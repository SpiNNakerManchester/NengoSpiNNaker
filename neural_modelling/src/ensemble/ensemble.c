// standard includes
#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <recording.h>
#include <bit_field.h>
#include <out_spikes.h>

// Ensemble includes
#include <ensemble/ensemble.h>
#include <ensemble/ensemble_common/neuron_lif.h>
#include <ensemble/ensemble_common/pes.h>
#include <ensemble/ensemble_common/voja.h>

// common includes
#include <common/packet_queue.h>
#include <common/fixed_point.h>
#include <common/input_filtering.h>
#include <common/nengo_typedefs.h>

// declare spin1_wfi
void spin1_wfi();

//! invalud recording region idex
#define INVALID_RECORDING_INDEX 255

//! number opf recording regions when using decoders as well
#define N_RECORDINGS_WITH_DECODERS 4

//! the initial learnt vector index to set off the state machine.
#define INITIAL_LEARNT_VECTOR_INDEX 0

//! offset from the input semaphore to ensure easier debugging
#define SEMAPHORE_SPIKE_OFFSET 3

//! accum set to 0.0
#define ZERO_ACCUM_CONSTANT 0.0k

//! bits in a word
#define BITS_IN_WORD 32

//! number of items used for each learnt input vector
#define SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR 2

//! enum mapping region ids to regions from python
typedef enum regions {
    SYSTEM = 0, ENSEMBLE_PARAMS = 1, ENCODER = 2, BIAS = 3, GAIN = 4,
    DECODER = 5, LEARNT_DECODER = 6, KEYS = 7, FILTERS = 8, ROUTING = 9,
    PES = 10, VOJA = 11, RECORDING_INDEXES = 12, RECORDING = 13
} regions;

//! enum mapping ensemble params in sdram from python
typedef enum ensemble_params_region_elements {
    START_ENSEMBLE_PARAMS = 0, START_LEARNT_INPUT_SIGNALS = 16
} ensemble_params_region_elements;

//! enum mapping recording index points in sdram from python
typedef enum recording_region_index_positions {
    N_RECORDING_VARIABLES = 0, RECORD_SPIKES_INDEX = 1, RECORD_VOLTAGE_INDEX
    = 2, RECORD_ENCODERS_INDEX = 3, RECORD_DECODERS_INDEX = 4
} recording_region_index_positions;

//! callback priorities
typedef enum callback_priorities{
    SDP = 1, TIMER = 2, DMA = 0, MCPL = -1, USER=1
} callback_priorities;

//! enum mapping for operation codes for DMA tags
typedef enum dma_operation_codes{
    // Write subspace of input into SDRAM
    WRITE_FILTERED_VECTOR,

    // Write subspace of learnt signal into SDRAM
    WRITE_FILTERED_LEARNT_VECTOR,

    // Read input vector into DTCM
    READ_WHOLE_VECTOR,

    // Read learned vector into DTCM
    READ_WHOLE_LEARNED_VECTOR,

    // Write spike vector into SDRAM
    WRITE_SPIKE_VECTOR,

    // Read spike vector into DTCM for decoding
    READ_SPIKE_VECTOR
}dma_operation_codes;

// Input vector synchronisation
volatile uint8_t *sema_input;

// Spike vector synchronisation
volatile uint8_t *sema_spikes;

//! the state for the ensemble
ensemble_state_t ensemble;

//! Queue of multicast packets
packet_queue_t packets;

//! Flag indicating that packets are being processed
bool queue_processing;

//! the number of packet queue overflows
unsigned int queue_overflows;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//! the int that represents the bool for if the run is infinite or not.
static uint32_t infinite_run;

//! the time interval parameter CANT USE TIMER_COUNT as not stable
static uint32_t time;

//! The expected current clock tick of timer_1
static uint32_t expected_time;

//! The expected time to wait between spikes
static uint32_t time_between_spikes;

//! The random backoff between timer ticks to desynchronize sources
static uint32_t random_backoff_us;

//! Input filters and buffers for general inputs. Their outputs
//! are summed into accumulators which are used to drive the standard neural
//! input
if_collection_t input_filters;

//! Input filters and buffers for inhibitory inputs. Their outputs
//! are summed into accumulators which are used to drive the standard neural
//! input
if_collection_t inhibition_filters;

//! Input filters and buffers for modulatory signals. Their outputs are left
//! seperate for use by learning rules
if_collection_t modulatory_filters;

//! Input filters and buffers for signals to be encoded by learnt encoders.
//! Each output is encoded by a seperate encoder so these are also left
//! seperate
if_collection_t learnt_encoder_filters;

//! Our portion of the shared learnt input vector
value_t **sdram_learnt_input_vector_local;

//! Input vectors corresponding to each learnt input signals - these are copied
//! From host, but would be a pain to append to the ensemble_parameters_t
//! struct
value_t **sdram_learnt_input_vector_addresses;

//! Index of learnt vector currently being DMAd (applies to both
//! WRITE_FILTERED_LEARNT_VECTOR and READ_WHOLE_LEARNED_VECTOR tags)
uint32_t dma_learnt_vector = 0;

//! size of sdram to write spikes for this core
uint spikes_write_size;

//! The recording flags
static uint32_t recording_flags = 0;

//! the index for voltage recording
uint8_t voltage_recording_index = INVALID_RECORDING_INDEX;

//! the index for scaled encoders recording
uint8_t scaled_encoders_recording_index = INVALID_RECORDING_INDEX;

//! the index for the output recording
uint8_t output_recording_index = INVALID_RECORDING_INDEX;

//! the index for the spike recording
uint8_t spikes_recording_index = INVALID_RECORDING_INDEX;

//! the index for the decoders recording
uint8_t decoder_recording_index = INVALID_RECORDING_INDEX;

//! number of possible recording variables
uint n_recording_variables = 0;

//! flag to ensure out spikes finished dma before restarting
static uint32_t n_recordings_outstanding = 0;

//! The values of the recorded voltages
static uint16_t *voltage_recording_values;

//! \brief callback when recording dma for variables finished
void recording_done_callback() {
    n_recordings_outstanding -= 1;
}

//! \brief resume callback to set recording stuff back
void resume_callback(){
    recording_reset();
}

//! \brief Simulate neurons and slowly dribble a spike vector out into a
//! given array. This function will also apply any encoder learning rules.
//! \param[in] ensemble: State of the ensemble
//! \param[in] spikes: Spike vector in which to record spikes
void simulate_neurons(ensemble_state_t *ensemble, uint32_t *spikes) {

    // Extract parameters
    value_t *input = ensemble->input;
    value_t inhib_input = ensemble->inhibitory_input;
    uint32_t n_dims = ensemble->parameters.n_dims;
    uint32_t encoder_width = ensemble->parameters.encoder_width;

    // Bit to use to indicate that a neuron spiked
    uint32_t bit = (1 << 31);

    // Cache for local spike vector
    uint32_t local_spikes = 0x0;

    // Set up an array for storing the recorded variable values
    value_t voltage = 0;

    // Wait until recordings have completed, to ensure the recording space
    // can be re-written
    log_info("n outstanding recordings = %d", n_recordings_outstanding);
    while (n_recordings_outstanding > 0) {
        spin1_wfi();
    }

    // Update each neuron in turn
    for (uint32_t n = 0; n < ensemble->parameters.n_neurons; n++) {
        // Get this neuron's encoder vector
        value_t *encoder_vector = &ensemble->encoders[encoder_width * n];

        // Is this neuron in its refractory period
        bool in_refractory_period = neuron_refractory(n, ensemble->state);

        // Loop through learnt input signals and encoder slices
        uint32_t f = 0;
        uint32_t e = n_dims;
        value_t neuron_input = ZERO_ACCUM_CONSTANT;
        for(; f < ensemble->parameters.n_learnt_input_signals;
                f++, e += n_dims){
            // Get encoder vector for this neuron offset for correct learnt
            // encoder
            const value_t *learnt_encoder_vector = encoder_vector + e;

            // Record learnt encoders
            // **NOTE** idea here is that by interspersing these between
            // encoding operations, write buffer should have time to be
            // written out
            //recording_record(
            //    scaled_encoders_recording_index, &learnt_encoder_vector,
            //    sizeof(value_t) * n_dims);

            // If neuron's not in refractory period,
            // apply input encoded by learnt encoders
            if(!in_refractory_period){
                neuron_input += dot_product(
                    n_dims, learnt_encoder_vector, ensemble->learnt_input[f]);
            }
        }

        // If the neuron's in its refractory period, decrement the refractory
        // counter
        /*
        if (in_refractory_period){
            neuron_refractory_decrement(n, ensemble->state);
        }
        else
        {
            // Compute the neuron input, this is a combination of (a) the
            // bias, (b) the inhibitory input times the gain and (c) the
            // non-learnt encoded input.
            neuron_input += ensemble->bias[n];
            neuron_input += inhib_input * ensemble->gain[n];

            // If there are any static input filters
            // **YUCK** this potentially massive optimisation
            // could also extend to memory by not allocating the encoders
            if(input_filters.n_filters > 0) {
                neuron_input += dot_product(n_dims, encoder_vector, input);
            }

            // Perform the neuron update
            if (neuron_step(n, neuron_input, ensemble->state, &voltage)) {
                // The neuron fired, record the fact in the spike vector that
                // we're constructing.
                local_spikes |= bit;
                out_spikes_set_spike(n);

                // Update non-filtered Voja learning
                value_t** const_learnt_input = ensemble->learnt_input;
                voja_neuron_spiked(
                    encoder_vector, ensemble->gain[n], n_dims,
                    &modulatory_filters, (const value_t**) const_learnt_input);
            }
        }*/

        // Rotate the neuron firing bit and progress the spike vector
        // if necessary.
        /*bit >>= 1;
        if (bit == 0) {
            // Bit is out of range so reset it
            bit = (1 << 31);

            // Copy spikes into the spike vector#
            *spikes = local_spikes;

            // Point at the next word in the spike vector
            spikes++;

            // Reset the local spike vector
            // Reset the local spike vector
            local_spikes = 0x0;
        }
        voltage_recording_values[n] = voltage;*/
    }
/*
    // Copy any remaining spikes into the specified spike vector
    if (ensemble->parameters.n_neurons % BITS_IN_WORD) {

        // Copy spikes into the spike vector
        *spikes = local_spikes;
    }*/

    // Finish up the recording
    /*n_recordings_outstanding += 1;

    uint32_t size_in_bytes = get_bit_field_size(ensemble->parameters.n_neurons);
    out_spikes_record(
        spikes_recording_index, time,
        get_bit_field_size(ensemble->parameters.n_neurons),
        recording_done_callback);
    n_recordings_outstanding += 1;
    if (voltage_recording_index != INVALID_RECORDING_INDEX){
        recording_record_and_notify(
            voltage_recording_index, voltage_recording_values,
            sizeof(uint16_t) * ensemble->parameters.n_neurons,
            recording_done_callback);
    }*/
}


//! \brief Decode a spike train to produce a single value
//! \param[in] decoder: Decoder to use
//! \param[in] n_populations: Number of populations
//! \param[in] population_lengths: Length of the populations
//! \param[in] spikes: Spike vector
//! \return payload to send to next core
static value_t decode_spike_train(
    const uint32_t n_populations, const uint32_t *population_lengths,
    const value_t *decoder, const uint32_t *spikes) {
    log_info("decode spike train");
    // Resultant decoded value
    value_t output = ZERO_ACCUM_CONSTANT;

    // For each population
    for (uint32_t population = 0; population < n_populations; population++)
    {
        // Get the number of neurons in this population
        uint32_t pop_length = population_lengths[population];

        // While we have neurons left to process
        while (pop_length)
        {
            // Determine how many neurons are in the next word of the spike
            // vector.
            uint32_t n_neurons =
                (pop_length > BITS_IN_WORD) ? BITS_IN_WORD : pop_length;

            // Load the next word of the spike vector
            uint32_t data = *(spikes++);

            // Include the contribution from each neuron
            while (n_neurons)  // While there are still neurons left
            {
                //TODO WTF!
                // Work out how many neurons we can skip
                // XXX: The GCC documentation claims that `__builtin_clz(0)` is
                // undefined, but the ARM instruction it uses is defined such
                // that: CLZ 0x00000000 is 32
                uint32_t skip = __builtin_clz(data);

                // If `skip` is NOT less than `n` then there are either no
                // firing neurons left in the word (`skip` == 32) or the first
                // `1` in the word is beyond the range of bits we care about
                // anyway.
                if (skip < n_neurons) {
                    // Skip until we reach the next neuron which fired
                    decoder += skip;

                    // Decode the given neuron
                    output += *decoder;

                    // Prepare to test the neuron after the one we just
                    // processed.
                    decoder++;

                    // Also skip the neuron we just decoded
                    skip++;

                    // Reduce the number of neurons left
                    pop_length -= skip;

                    // and the number left in this word.
                    n_neurons -= skip;

                    // Shift out processed neurons
                    data <<= skip;
                }
                else{   // There are no neurons left in this word
                    // Point at the decoder for the next neuron
                    decoder += n_neurons;

                    // Reduce the number left in the population
                    pop_length -= n_neurons;

                    // No more neurons left to process
                    n_neurons = 0;
                }
            }
        }
    }
    // Return the decoded value
    return output;
}

//! \brief Apply the decoder to a spike vector and transmit multicast packets
//         representing the decoded vector.  This function will also apply any
//         decoder learning rules.
//! \param[in] ensemble: ensemble state params.
static inline void decode_output_and_transmit(
        const ensemble_state_t *ensemble){

    // Sleep for a random time so that packet transmission occurs some time
    // after the timer tick.  For shorter simulations this will hide the effect
    // of clock drift for a short period.
    log_info("a");
    spin1_delay_us(random_backoff_us);

    // Extract parameters
    log_info("b");
    const ensemble_parameters_t *params = &ensemble->parameters;
    log_info("c");
    uint32_t n_neurons_total = params->n_neurons_total;
    log_info("d");
    uint32_t n_populations = params->n_populations;
    log_info("e");
    uint32_t n_decoder_rows =
        params->n_decoder_rows + params->n_learnt_decoder_rows;

    log_info("f");
    uint32_t *pop_lengths = ensemble->population_lengths;
    log_info("g");
    value_t *decoder = ensemble->decoders;
    log_info("h");
    uint32_t *keys = ensemble->keys;
    log_info("i");
    uint32_t *spike_vector = ensemble->spikes;
    log_info("j");

    // Apply the decoder and transmit multicast packets.
    // Each decoder row is applied in turn to get the output value, which is then
    // transmitted.
    log_info("loop");
    for (uint32_t d = 0; d < n_decoder_rows; d++) {

        log_info("k");
        // Get the row of the decoder
        value_t *row = &decoder[d * n_neurons_total];

        // Compute the decoded value
        log_info("l");
        value_t output = decode_spike_train(n_populations, pop_lengths,
                                            row, spike_vector);

        // Wait until the expected time to send
        log_info("m");
        while (tc[T1_COUNT] > expected_time) {
            // Do Nothing
        }
        expected_time -= time_between_spikes;

        // Transmit this value (keep trying until it sends)
        log_info("n");
        while(!spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD)){
            spin1_delay_us(1);
        }
        log_info("o");
    }
    log_info("p");
}

//! \brief Multicast packet with payload received. Puts packet into queue and
//!        sets off user event.
//! \param[in] key: the mc key
//! \param[in] payload: the mc payload
void multicast_payload_callback(uint key, uint payload) {
    log_info("packet!");
    // Queue the packet for later processing, if no processing is scheduled then
    // trigger the queue processor.
    if (packet_queue_push(&packets, key, payload)) {
        if (!queue_processing) {
            spin1_trigger_user_event(0, 0);
            queue_processing = true;
        }
    }
    else {
        // The packet couldn't be included in the queue, thus it was essentially
        // dropped.
        queue_overflows++;
    }
    log_info("packet Done !");
}

//! \brief extracts packets from queue and applies input filters
void process_queue() {
    log_info("process queue !");
    uint32_t offset = ensemble.parameters.input_subspace_offset;
    uint32_t max_dim_sub_one = ensemble.parameters.input_subspace_n_dims - 1;

    // Continuously remove packets from the queue and include them in filters
    while (packet_queue_not_empty(&packets))
    {
        // Pop a packet from the queue (critical section)
        packet_t packet;
        uint cpsr = spin1_fiq_disable();
        bool packet_is_valid = packet_queue_pop(&packets, &packet);
        spin1_mode_restore(cpsr);

        // Process the received packet
        if (packet_is_valid)
        {
            uint32_t key = packet.key;
            uint32_t payload = packet.payload;

            // Standard input
            input_filtering_input_with_dimension_offset(
                &input_filters, key, payload, offset, max_dim_sub_one);

            // Learnt encoder input
            input_filtering_input_with_dimension_offset(
                &learnt_encoder_filters, key, payload, offset,
                max_dim_sub_one);

            // Inhibitory
            input_filtering_input(&inhibition_filters, key, payload);

            // Modulatory
            input_filtering_input(&modulatory_filters, key, payload);
        }
        else
        {
            log_error("Popped packet from empty queue.\n");
            rt_error(RTE_ABORT);
        }
    }
    queue_processing = false;
}

//! \brief pulls the packets out of the queue.
//! \param[in] arg1: meh
//! \param[in] arg0: meh
void user_event(uint arg0, uint arg1) {
    log_info("user");
    use(arg0);
    use(arg1);
    process_queue();
}

//! \brief Writes the accumulated local section of the standard filter's
//!        output to SDRAM
static inline void write_filtered_vector() {
    log_info("write filtered vector");
    // Compute the size of the transfer
    uint size = sizeof(value_t) * ensemble.parameters.input_subspace_n_dims;

    spin1_dma_transfer(
        WRITE_FILTERED_VECTOR, ensemble.parameters.sdram_input_vector_local,
        ensemble.input_local, DMA_WRITE, size);
}

//! \brief Writes the local section of the specified learnt input filter's
//!        output to SDRAM
//! \param[in] signal: the index for the next learnt vector to process
static inline void write_filtered_learnt_vector(uint32_t signal) {
    log_info("write filtered learnt vector");
    // Cache index
    dma_learnt_vector = signal;

    // Compute the size of the transfer
    uint size = sizeof(value_t) * ensemble.parameters.input_subspace_n_dims;

    spin1_dma_transfer(
        WRITE_FILTERED_LEARNT_VECTOR, sdram_learnt_input_vector_local[signal],
        ensemble.learnt_input_local[signal], DMA_WRITE, size);
}

//! \brief Reads whole input vector from SDRAM
static inline void read_whole_vector() {
    log_info("read whole vector");
    // Schedule reading in the whole input vector from SDRAM
    value_t *sdram_input_vector = ensemble.parameters.sdram_input_vector;
    spin1_dma_transfer(
        READ_WHOLE_VECTOR, sdram_input_vector, ensemble.input, DMA_READ,
        sizeof(value_t) * ensemble.parameters.n_dims);
}

//! \brief Reads whole input vector from SDRAM
//! \param[in] signal: the index for the next learnt vector to read in
static inline void read_whole_learned_vector(uint32_t signal) {
    log_info("read whole learned vector");
    // Cache index
    dma_learnt_vector = signal;

    spin1_dma_transfer(
        READ_WHOLE_LEARNED_VECTOR, sdram_learnt_input_vector_addresses[signal],
        ensemble.learnt_input[signal], DMA_READ,
        sizeof(value_t) * ensemble.parameters.n_dims);
}

//! \brief applies the pes rules to the modulatory filters
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_read_spike_vector(uint transfer_id, uint tag){
    log_info("read spike vector");
    use(transfer_id);
    use(tag);

    // Apply PES learning to spike vector
    pes_apply(&ensemble, &modulatory_filters);

    // Decode and transmit neuron output
    decode_output_and_transmit(&ensemble);
}

//! \brief wait for the sdram edge for spike vector to complete and then read
//!        back in
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_write_spike_vector(uint transfer_id, uint tag){
    log_info("write spike vector");
    use(transfer_id);
    use(tag);

    // Wait for all cores to have written their spike vectors into SDRAM
    sark_sema_lower((uchar *) sema_spikes);
    while (*sema_spikes){
        // do nothing. busy wait for the sdram edge to complete
    }

    // Schedule reading back the whole spike vector from SDRAM
    uint32_t *sdram_spike_vector = ensemble.parameters.sdram_spike_vector;
    spin1_dma_transfer(
        READ_SPIKE_VECTOR, sdram_spike_vector, ensemble.spikes, DMA_READ,
        sizeof(uint32_t) * ensemble.sdram_spikes_length);
}

//! \brief cycles vectors
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_read_whole_learnt_vector(uint transfer_id, uint tag){
    log_info("read whole learnt vector");
    use(transfer_id);
    use(tag);

    // Get index of next learnt vector
    uint32_t next_learnt_vector = dma_learnt_vector + 1;

    // If there are more to go, read next learnt vector from sdram
    if(next_learnt_vector < ensemble.parameters.n_learnt_input_signals) {
        read_whole_learned_vector(next_learnt_vector);
    }
    else{  // Otherwise, read whole vector
        read_whole_vector();
    }
}

//! \brief simulates the neurons of the ensamble and then writes the spike
//!        vector to sdram
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_read_whole_vector(uint transfer_id, uint tag){
    log_info("read whole vector");
    use(transfer_id);
    use(tag);

    // Process the neurons, then copy the spike vector into SDRAM.
    simulate_neurons(&ensemble, ensemble.spikes);

    // Schedule writing out the spike vector
    spin1_dma_transfer(
        WRITE_SPIKE_VECTOR,         // Tag
        ensemble.parameters.sdram_spikes_vector_local,  // SDRAM address
        ensemble.spikes,            // DTCM addess
        DMA_WRITE,                  // Direction
        spikes_write_size);
}

//! \brief writes next filtered learnt vector or filtered learnt vector
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_write_filtered_learnt_vector(uint transfer_id, uint tag){
    log_info("write filtered learnt vector");
    use(transfer_id);
    use(tag);

    // Get index of next learnt vector
    uint32_t next_learnt_vector = dma_learnt_vector + 1;

    // If there are more to go, write next learnt vector to sdram
    if(next_learnt_vector < ensemble.parameters.n_learnt_input_signals) {
        write_filtered_learnt_vector(next_learnt_vector);
    }
    else { // Otherwise, write filtered vector
        write_filtered_vector();
    }
}

//! \brief  waits for the sdram edge for filtered vector to complete. then reads
//!         all back in
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_write_filtered_vector(uint transfer_id, uint tag){
    log_info("write filtered vector");
    use(transfer_id);
    use(tag);

    // Wait for all cores to have written their input vectors into SDRAM
    sark_sema_lower((uchar *) sema_input);
    while (*sema_input){
        // do nothing. waiting for sdram vector to complete
    }

    // If there are any learnt input signals to transfer, start reading of 1st
    // signal
    if(ensemble.parameters.n_learnt_input_signals > 0) {
        read_whole_learned_vector(0);
    }
    else{
        read_whole_vector();
    }
}

//! \brief On every timer tick the ensemble executable should apply filtering
//!        to the portion of the subspace that it is responsible and copy its
//!        filtered input into the shared input in SDRAM.
//! \param[in] unused: nah
//! \param[in] timer_count: not useful tracker for how many times the timer tick
//!                         inturrupt has been called
void timer_callback(uint timer_count, uint unused) {
    use(timer_count);
    use(unused);
    time++;

    log_info("Timer tick %u", time);

    // If a fixed number of simulation ticks are specified and these have
    // passed
    if (infinite_run != TRUE && time >= simulation_ticks) {

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

        // Finalise any recordings that are in progress, writing back the final
        // amounts of samples recorded to SDRAM
        if (recording_flags > 0) {
            log_debug("updating recording regions");
            recording_finalise();
        }

        // Subtract 1 from the time so this tick gets done again on the next
        // run
        time -= 1;
        simulation_ready_to_read();
        return;
    }

    // If there are multiple cores for this ensemble then raise the
    // synchronisation semaphores
    if (ensemble.parameters.n_populations > 1) {
        sark_sema_raise((uchar *) sema_input);
        sark_sema_raise((uchar *) sema_spikes);
    }

    // extract packets
    process_queue();

    // process input filters
    log_info("process input filters!");
    input_filtering_step(&input_filters);
    log_info("process inhib filters!");
    input_filtering_step(&inhibition_filters);
    log_info("process mod filters!");
    input_filtering_step_no_accumulate(&modulatory_filters);
    log_info("process learnt filters!");
    input_filtering_step_no_accumulate(&learnt_encoder_filters);

    // If there are multiple cores for this ensemble then schedule copying the
    // input vector into SDRAM.  Otherwise if this is address is NULL start
    // processing the neurons.
    log_info("possibley process write filtered!");
    if (ensemble.parameters.n_populations > 1) {
        // If there are any learnt input signals to transfer, start transfer of 1st signal
        log_info("process write filtered!");
        if(ensemble.parameters.n_learnt_input_signals > 0) {
            write_filtered_learnt_vector(INITIAL_LEARNT_VECTOR_INDEX);
        }
        else { // Otherwise, transfer our section of filtered vector
            write_filtered_vector();
        }
    }
    else {
        log_info("start sim neurons!");
        // Process the neurons, writing the spikes out into DTCM rather than a
        // shared SDRAM vector.
        simulate_neurons(&ensemble, ensemble.spikes);

        // Apply PES learning to spike vector
        //log_info("start pes apply");
        //pes_apply(&ensemble, &modulatory_filters);

        // Decode and transmit output
        //log_info("decode and trasnmit");
        //decode_output_and_transmit(&ensemble);
    }
    log_info("doen timer");

}

//! \brief initialises the ensemble_parameters_t struct, and the learnt input
//!        vectors.
//! \param[in] region_address: the sdram address for the start of the ensemble
//!                            data
//! \return bool that states if the init was successful
static bool ensemble_param_read(address_t region_address){
    // Copy in the ensemble parameters into the ensemble params holder
    ensemble_parameters_t *params = &ensemble.parameters;

    // read in the params from sdram
    spin1_memcpy(
        params,
        &region_address[START_ENSEMBLE_PARAMS],
        sizeof(ensemble_parameters_t));

    // update the semaphore sdram addresses
    uint8_t *sem_base = (uint8_t *) params->semaphore_base_address;
    sema_input = sem_base;
    sema_spikes = sem_base + SEMAPHORE_SPIKE_OFFSET;

    // process ensemble input
    ensemble.input = spin1_malloc(sizeof(value_t) * params->n_dims);
    if (ensemble.input == NULL){
        log_error("cannot allocate dtcm for the ensemble input");
        return false;
    }

    // Allocate array to hold pointers to SDRAM learnt input vectors
    log_info("allocating for %d learnt input signals",
             params->n_learnt_input_signals);
     uint sdram_pointer = START_LEARNT_INPUT_SIGNALS;
     if (params->n_learnt_input_signals != 0){
        sdram_learnt_input_vector_addresses = spin1_malloc(
            sizeof(value_t*) * params->n_learnt_input_signals);
        if(sdram_learnt_input_vector_addresses == NULL){
            log_error("Cannot allocate dtcm for the learnt input signals");
            return false;
        }

        // Allocate an array of local, global and shared pointers for each
        // learnt input signals
        sdram_learnt_input_vector_local = spin1_malloc(
            sizeof(value_t*) * params->n_learnt_input_signals);
        if(sdram_learnt_input_vector_local == NULL){
            log_error(
                "cannot allocate dtcm for the local learnt input signals");
            return false;
        }

        // copy in pointers for learnt input vector sdram (local and global)
        uint learnt_vector_local_pointer = START_LEARNT_INPUT_SIGNALS + 1;
        for (uint32_t learnt_input_signal=0;
                learnt_input_signal < params->n_learnt_input_signals;
                learnt_input_signal ++) {
            sdram_learnt_input_vector_addresses[learnt_input_signal] =
                (value_t*) &region_address[sdram_pointer];
            sdram_learnt_input_vector_local[learnt_input_signal] =
                (value_t*) &region_address[learnt_vector_local_pointer];
            sdram_pointer += SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR;
            learnt_vector_local_pointer += SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR;
        }


        // allocate learnt input space
        ensemble.learnt_input = spin1_malloc(
            sizeof(value_t*) * params->n_learnt_input_signals);
        if (ensemble.learnt_input == NULL){
            log_error("Cannot allocate dtcm for the ensemble learnt input");
            return false;
        }

        ensemble.learnt_input_local = spin1_malloc(
            sizeof(value_t*) * params->n_learnt_input_signals);
        if (ensemble.learnt_input_local == NULL){
            log_error(
                "cannot allocate dtcm for the ensemble learnt input local");
            return false;
        }

        // Loop through learnt input signals and read in as needed
        for(uint32_t input_signal = 0;
                input_signal < params->n_learnt_input_signals; input_signal++) {
            ensemble.learnt_input[input_signal] = spin1_malloc(
                sizeof(value_t) * params->n_dims);
            if (ensemble.learnt_input[input_signal] == NULL){
                log_error(
                    "failed to allocate dtcm for learnt input %d",
                    input_signal);
                return false;
            }

            // Store local offset
            ensemble.learnt_input_local[input_signal] =
                &ensemble.learnt_input[input_signal][
                    params->input_subspace_offset];

            log_debug(
                "Learnt input signal %u: learnt_input:%08x, "
                "learnt_input_local:%08x, sdram_learnt_input:%08x, "
                "sdram_learnt_input_local:%08x\n", input_signal,
                ensemble.learnt_input[input_signal],
                ensemble.learnt_input_local[input_signal],
                sdram_learnt_input_vector_addresses[input_signal],
                sdram_learnt_input_vector_local[input_signal]);
        }
    }

    // read in the lif params
    uint32_t words_read = 0;
    lif_prepare_state(
        &ensemble, &region_address[sdram_pointer], &words_read);
    sdram_pointer += words_read;

    // read in pop length data
    uint poplength_size = sizeof(uint32_t) * params->n_populations;
    ensemble.population_lengths = spin1_malloc(poplength_size);
    if (ensemble.population_lengths == NULL){
        log_error("cannot allocate dtcm for ensmeble population lengths");
        return false;
    }
    spin1_memcpy(
        ensemble.population_lengths, &region_address[sdram_pointer],
        poplength_size);

    return true;
}

//! \brief sets up the spike write size
void set_spike_write_size() {
    spikes_write_size = get_bit_field_size(ensemble.parameters.n_neurons);
    // Compute the spike size for writing into SDRAM
    spikes_write_size = ensemble.parameters.n_neurons / BITS_IN_WORD;
    if (ensemble.parameters.n_neurons % BITS_IN_WORD) {
        spikes_write_size++;
    }
    spikes_write_size *= sizeof(uint32_t);
}

//! \brief reads in the filters for the ensemble. this includes the following
//!        filters : INPUT, INHIB, MODULATORY, LEARNT
//! \param[in] region_address: the sdram address for the start of the ensemble
//!                            data
//! \return bool that states if the init was successful
bool ensemble_setup_filters(address_t address){
    uint32_t words_read = 0;
    uint32_t total_words_read = 0;

    // process input filters
    log_info("sorting out input filters");
    if(!input_filtering_initialise_filters(
            &input_filters, address, NULL, &words_read)){
        return false;
    }
    total_words_read += words_read;

    // final input filter updating
    input_filters.output_size = ensemble.parameters.input_subspace_n_dims;
    input_filters.output = ensemble.input_local;

    // process inhib filters
    log_info("sorting out inhib filters");
    if(!input_filtering_initialise_filters(
            &inhibition_filters, &address[total_words_read], NULL,
            &words_read)){
        return false;
    }
    total_words_read += words_read;

    // final inhib filter updating
    inhibition_filters.output_size = 1;
    inhibition_filters.output = &ensemble.inhibitory_input;


    // process modulatory filters
    log_info("sorting out modulatory filters");
    if(!input_filtering_initialise_filters(
            &modulatory_filters, &address[total_words_read], NULL,
            &words_read)){
        return false;
    }
    total_words_read += words_read;


    // process learnt encoder filters
    log_info("sorting out learnt encoders filters");
    if(!input_filtering_initialise_filters(
            &learnt_encoder_filters, &address[total_words_read],
            ensemble.learnt_input_local, &words_read)){
        return false;
    }

    // report success
    return true;
}

//! \brief reads in the routes for the ensemble. this includes the following
//!        routes : INPUT, INHIB, MODULATORY, LEARNT
//! \param[in] region_address: the sdram address for the start of the ensemble
//!                            data
//! \return bool that states if the init was successful
bool ensemble_setup_routes(address_t address){
    uint32_t words_read = 0;
    uint32_t total_words_read = 0;

    // process input filters
    if(!input_filtering_initialise_routes(
            &input_filters, address, &words_read)){
        return false;
    }
    log_info("read %d words", words_read);
    total_words_read += words_read;
    words_read = 0;

    // process inhib filters
    if(!input_filtering_initialise_routes(
        &inhibition_filters, &address[total_words_read], &words_read)){
        return false;
    }
    log_info("read %d words", words_read);
    total_words_read += words_read;
    words_read = 0;

    // process modulatory filters
    if(!input_filtering_initialise_routes(
        &modulatory_filters, &address[total_words_read], &words_read)){
        return false;
    }
    log_info("read %d words", words_read);
    total_words_read += words_read;
    words_read = 0;

    // process learnt encoder filters
    if(!input_filtering_initialise_routes(
        &learnt_encoder_filters, &address[total_words_read], &words_read)){
        return false;
    }
    log_info("read %d words", words_read);

    // report success
    return true;
}

//! \brief reads in all matrix based regions from python (
//!        encoder, bias, gain, decoders and learnt decoders)
//! \param[in] dsg_address: the address to the dsg pointer table.
//! \return bool stating if the setup was successful.
bool ensemble_setup_matrix_based_regions(address_t dsg_address){
    // Copy in encoders
    uint encoder_size =
        sizeof(value_t) * ensemble.parameters.n_neurons *
        ensemble.parameters.encoder_width;
    ensemble.encoders = spin1_malloc(encoder_size);
    if (ensemble.encoders == NULL){
        log_error("failed to allocate DTCM for ensemble encoders");
        return false;
    }
    spin1_memcpy(
        ensemble.encoders, data_specification_get_region(ENCODER, dsg_address),
        encoder_size);

    // copy in bias
    uint bias_size = sizeof(value_t) * ensemble.parameters.n_neurons;
    ensemble.bias = spin1_malloc(bias_size);
    if (ensemble.bias == NULL){
        log_error("failed to allocate DTCM for ensemble bias");
        return false;
    }
    spin1_memcpy(
        ensemble.bias, data_specification_get_region(BIAS, dsg_address),
        bias_size);

    // copy in gain
    uint gain_size = sizeof(value_t) * ensemble.parameters.n_neurons;
    ensemble.gain = spin1_malloc(gain_size);
    if (ensemble.gain == NULL){
        log_error("failed to allocate DTCM for ensemble gain");
        return false;
    }
    spin1_memcpy(
        ensemble.gain, data_specification_get_region(GAIN, dsg_address),
        gain_size);

    // copy in decoders and learnt decoders
    const uint32_t decoder_words =
        ensemble.parameters.n_neurons_total *
        ensemble.parameters.n_decoder_rows;
    const uint32_t learnt_decoder_words =
        ensemble.parameters.n_neurons_total *
        ensemble.parameters.n_learnt_decoder_rows;

    ensemble.decoders = spin1_malloc(
        (decoder_words + learnt_decoder_words) * sizeof(value_t));
    if (ensemble.decoders == NULL){
        log_error("failed to allocate DTCM for ensemble decoders and learnt "
                  "decoders");
        return false;
    }
    spin1_memcpy(
        ensemble.decoders, data_specification_get_region(DECODER, dsg_address),
        decoder_words * sizeof(value_t));
    spin1_memcpy(
        ensemble.decoders + decoder_words,
        data_specification_get_region(LEARNT_DECODER, dsg_address),
        learnt_decoder_words * sizeof(value_t));
    return true;
}

//! \brief Initialises the recording parts of the model
//! \param[in] recording_address: the address in SDRAM where to store
//! recordings
//! \return True if recording initialisation is successful, false otherwise
static bool initialise_recording(address_t recording_address){
    bool success = recording_initialize(recording_address, &recording_flags);
    log_debug("Recording flags = 0x%08x", recording_flags);
    return success;
}

//! \brief reads in the recording index for the model
//! \param[in] recording_index_address: the address in SDRAM where to store
//! recording indexes
//! \return True if recording initialisation is successful, false otherwise
static bool read_in_recording_indexs(address_t recording_index_address){
    uint32_t n_recording_regions =
        recording_index_address[N_RECORDING_VARIABLES];


    // read indexes from sdram
    voltage_recording_index =
        (uint8_t) recording_index_address[RECORD_VOLTAGE_INDEX];
    scaled_encoders_recording_index =
        (uint8_t) recording_index_address[RECORD_ENCODERS_INDEX];
    spikes_recording_index =
        (uint8_t) recording_index_address[RECORD_SPIKES_INDEX];

    if (n_recording_regions == N_RECORDINGS_WITH_DECODERS){
        decoder_recording_index =
            (uint8_t) recording_index_address[RECORD_DECODERS_INDEX];
    }
    else{
        decoder_recording_index = INVALID_RECORDING_INDEX;
    }

    log_info(
        "volt record id %d scaled encoder record id %d spikes recording id %d "
        "decoder recording id %d",
         voltage_recording_index, scaled_encoders_recording_index,
         spikes_recording_index, decoder_recording_index);

    // instantiate the recording stores for voltages
    voltage_recording_values = (uint16_t *) spin1_malloc(
        sizeof(uint16_t) * ensemble.parameters.n_neurons);
    if (voltage_recording_values == NULL){
        log_error("could not allocate dtcm for voltage recording");
        return false;
    }

    // set up recording for decoders
    //TODO THIS NEEDS DOING IF WE DECIDE TO REMOVE PROBES

    // Set up the out spikes array - this is always n_neurons in size to ensure
    // it continues to work if changed between runs, but less might be used in
    // any individual run
    if (!out_spikes_initialize(ensemble.parameters.n_neurons)) {
        return false;
    }
    return true;
}

//! Initialises the model by reading in the regions and checking recording
//! data.
//! \param[out] timer_period a pointer for the memory address where the timer
//!            period should be stored during the function.
//! \return boolean of True if it successfully read all the regions and set up
//!         all its internal data structures. Otherwise returns False
static bool initialize(uint32_t *timer_period){
    log_info("Initialise: started");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address)) {
        return false;
    }

    // Get the timing details and set up the simulation interface
    log_info("sorting out simulation init");
    if (!simulation_initialise(
        data_specification_get_region(SYSTEM, address),
        APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
        &infinite_run, SDP, DMA)) {
        return false;
    }

    // get the ensemble params from sdram
    log_info("sorting out ensemble param init");
    if (!ensemble_param_read(
            data_specification_get_region(ENSEMBLE_PARAMS, address))){
        return false;
    }

    // set up the spikes write size
    log_info("sorting out spike write size");
    set_spike_write_size();

    // set up filters for the different filter types
    log_info("sorting out filters");
    if (!ensemble_setup_filters(
            data_specification_get_region(FILTERS, address))) {
        return false;
    }

    // set up routes for the different filters/routes types
    log_info("sorting out routes");
    if (!ensemble_setup_routes(
            data_specification_get_region(ROUTING, address))){
        return false;
    }

    // sort out matrix reads for encoder, bias, gain, decoders
    log_info("sorting out setup matrix");
    if (!ensemble_setup_matrix_based_regions(address)){
        return false;
    }

    // sort out pes learning rules
    log_info("sorting out pes");
    if (!pes_initialise(data_specification_get_region(PES, address))){
        return false;
    }

    // sort out voja learning rules
    log_info("sorting out voja");
    if (!voja_initialise(data_specification_get_region(VOJA, address))){
        return false;
    }

    // sort out recording region
    log_info("sorting out recording");
    if (!initialise_recording(
            data_specification_get_region(RECORDING, address))){
        return false;
    }

    // sort out recording index's
    log_info("sorting out recording index");
    if (!read_in_recording_indexs(
            data_specification_get_region(RECORDING_INDEXES, address))) {
        return false;
    }

    return true;
}

//! \brief entry method
void c_main(void){
    // Load DTCM data
    uint32_t timer_period;
    if (!initialize(&timer_period)) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick
    spin1_set_timer_tick(timer_period);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    spin1_callback_on(MCPL_PACKET_RECEIVED, multicast_payload_callback, MCPL);
    spin1_callback_on(USER_EVENT, user_event, USER);

    // register all the dma complete callbacks (creates state machine)
    simulation_dma_transfer_done_callback_on(
        WRITE_FILTERED_VECTOR, dma_complete_write_filtered_vector);
    simulation_dma_transfer_done_callback_on(
        WRITE_FILTERED_LEARNT_VECTOR,
        dma_complete_write_filtered_learnt_vector);
    simulation_dma_transfer_done_callback_on(
        READ_WHOLE_VECTOR, dma_complete_read_whole_vector);
    simulation_dma_transfer_done_callback_on(
        READ_WHOLE_LEARNED_VECTOR, dma_complete_read_whole_learnt_vector);
    simulation_dma_transfer_done_callback_on(
        WRITE_SPIKE_VECTOR, dma_complete_write_spike_vector);
    simulation_dma_transfer_done_callback_on(
        READ_SPIKE_VECTOR, dma_complete_read_spike_vector);

    // set up run
    simulation_run();
}