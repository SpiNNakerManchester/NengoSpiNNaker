// standard includes
#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <common/nengo_typedefs.h>

// Ensemble includes
#include <ensemble/ensemble_common/filtered_activity.h>
#include <ensemble/ensemble_common/neuron_lif.h>
#include <ensemble/ensemble_common/pes.h>
#include <ensemble/ensemble_common/voja.h>

// common includes
#include <common/packet_queue.h>
#include <common/fixed_point.h>
#include <common/input_filtering.h>

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

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, ENSEMBLE_PARAMS, NEURON, ENCODER, BIAS, GAIN, DECODER,
    LEARNT_DECODER, KEYS, FILTERS, ROUTING, PES, VOJA, FILTERED_ACTIVITY,
    RECORDING
} regions;

//! enum mapping ensemble params in sdram in python
typedef enum ensemble_params_region_elements {
    START_ENSEMBLE_PARAMS = 0, START_LEARNT_INPUT_SIGNALS = 16}

//! enum mapping ensemble params in sdram from python after learnt encoders
typedef enum ensemble_params_region_elements_after_learnt_encoders_addresses {
    exp_dt_over_tau_rc
}

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

//! Parameters for the locally represented neurons this is all data stored
//! within the system region.
typedef struct _ensemble_parameters
{
    // Number of neurons in this portion
    uint32_t n_neurons;

    // Number of dimensions represented
    uint32_t n_dims;

    // Total width of encoder
    uint32_t encoder_width;

    // Number of neurons overall
    uint32_t n_neurons_total;

    // Number of neurons overall
    uint32_t n_populations;

    // Index of this population
    uint32_t population_id;

    // Index of first dimension
    uint32_t input_subspace_offset;

    // Number of dimensions
    uint32_t input_subspace_n_dims;

    // Number of output dimensions
    uint32_t n_decoder_rows;

    // Number of learnt output dimensions
    uint32_t n_learnt_decoder_rows;

    // Pointer into SDRAM for the input vector
    value_t *sdram_input_vector;

    // Our portion of the shared input vector
    value_t *sdram_input_vector_local;

    // Pointer into SDRAM for the spike vector
    uint32_t *sdram_spike_vector;

    // Our portion of the shared spike vector
    uint32_t *sdram_spikes_vector_local;

    // semaphore base sdram address
    uint32_t semaphore_base_address;

    // Number of learnt input signals
    uint32_t n_learnt_input_signals

} ensemble_parameters_t;

//! state for the ensemble
typedef struct _ensemble_state
{
    // Generic parameters
    ensemble_parameters_t parameters;

    // Neuron state
    void *state;

    // Filtered input vector
    value_t *input;

    // Start of the section of input we update
    value_t *input_local;

    // Filtered input vector from each signal
    value_t **learnt_input;

    // Start of section of learnt_input we update
    value_t **learnt_input_local;

    // Globally inhibitory input
    value_t inhibitory_input;

    // Encoder matrix
    value_t *encoders;

    // Neuron biases
    value_t *bias;

    // Neuron gains
    value_t *gain;

    // Lengths of all populations
    uint32_t *population_lengths;

    // Length of padded spike vector (words)
    uint32_t sdram_spikes_length;

    // Unpadded spike vector
    uint32_t *spikes;

    // Rows from the decoder matrix
    value_t *decoders;

    // Output keys
    uint32_t *keys;
} ensemble_state_t;

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
            record_learnt_encoders(
                &record_encoders, n_dims, learnt_encoder_vector);

            // If neuron's not in refractory period,
            // apply input encoded by learnt encoders
            if(!in_refractory_period){
                neuron_input += dot_product(
                    n_dims, learnt_encoder_vector, ensemble->learnt_input[f]);
            }
        }

        // If the neuron's in its refractory period, decrement the refractory
        // counter
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
            if (neuron_step(
                    n, neuron_input, ensemble->state, &record_voltages)) {
                // The neuron fired, record the fact in the spike vector that
                // we're constructing.
                local_spikes |= bit;
                record_spike(&record_spikes, n);

                // Apply effect of neuron spiking to filtered activities
                //filtered_activity_neuron_spiked(n);

                // Update non-filtered Voja learning
                voja_neuron_spiked(
                    encoder_vector, ensemble->gain[n], n_dims,
                    &modulatory_filters, ensemble->learnt_input);
            }
        }

        // Rotate the neuron firing bit and progress the spike vector
        // if necessary.
        bit >>= 1;
        if (bit == 0) {
            // Bit is out of range so reset it
            bit = (1 << 31);

            // Copy spikes into the spike vector
            *spikes = local_spikes;

            // Point at the next word in the spike vector
            spikes++;

            // Reset the local spike vector
            local_spikes = 0x0;
        }
    }

    // Copy any remaining spikes into the specified spike vector
    if (ensemble->parameters.n_neurons % BITS_IN_WORD) {

        // Copy spikes into the spike vector
        *spikes = local_spikes;
    }

    // Finish up the recording
    record_buffer_flush(&record_voltages);
    record_buffer_flush(&record_spikes);
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
    spin1_delay_us(random_backoff_us);

    // Extract parameters
    const ensemble_parameters_t *params = &ensemble->parameters;
    uint32_t n_neurons_total = params->n_neurons_total;
    uint32_t n_populations = params->n_populations;
    uint32_t n_decoder_rows =
        params->n_decoder_rows + params->n_learnt_decoder_rows;

    uint32_t *pop_lengths = ensemble->population_lengths;
    value_t *decoder = ensemble->decoders;
    uint32_t *keys = ensemble->keys;
    uint32_t *spike_vector = ensemble->spikes;

    // Apply the decoder and transmit multicast packets.
    // Each decoder row is applied in turn to get the output value, which is then
    // transmitted.
    for (uint32_t d = 0; d < n_decoder_rows; d++) {
        // Get the row of the decoder
        value_t *row = &decoder[d * n_neurons_total];

        // Compute the decoded value
        value_t output = decode_spike_train(n_populations, pop_lengths,
                                            row, spike_vector);

        // Wait until the expected time to send
        while (tc[T1_COUNT] > expected_time) {
            // Do Nothing
        }
        expected_time -= time_between_spikes;

        // Transmit this value (keep trying until it sends)
        while(!spin1_send_mc_packet(keys[d], bitsk(output), WITH_PAYLOAD)){
            spin1_delay_us(1);
        }
    }
}

//! \brief Multicast packet with payload received. Puts packet into queue and
//!        sets off user event.
//! \param[in] key: the mc key
//! \param[in] payload: the mc payload
void mcpl_received(uint key, uint payload) {
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
}

//! \brief extracts packets from queue and applies input filters
void process_queue() {

    uint32_t offset = ensemble.parameters.input_subspace.offset;
    uint32_t max_dim_sub_one = ensemble.parameters.input_subspace.n_dims - 1;

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
    use(arg0);
    use(arg1);
    process_queue();
}

//! \brief Writes the accumulated local section of the standard filter's
//!        output to SDRAM
static inline void write_filtered_vector() {
    // Compute the size of the transfer
    uint size = sizeof(value_t) * ensemble.parameters.input_subspace.n_dims;

    spin1_dma_transfer(
        WRITE_FILTERED_VECTOR, sdram_input_vector_local,
        ensemble.input_local, DMA_WRITE, size);
}

//! \brief Writes the local section of the specified learnt input filter's
//!        output to SDRAM
//! \param[in] signal: the index for the next learnt vector to process
static inline void write_filtered_learnt_vector(uint32_t signal) {
    // Cache index
    dma_learnt_vector = signal;

    // Compute the size of the transfer
    uint size = sizeof(value_t) * ensemble.parameters.input_subspace.n_dims;

    spin1_dma_transfer(
        WRITE_FILTERED_LEARNT_VECTOR, sdram_learnt_input_vector_local[signal],
        ensemble.learnt_input_local[signal], DMA_WRITE, size);
}

//! \brief Reads whole input vector from SDRAM
static inline void read_whole_vector() {
    // Schedule reading in the whole input vector from SDRAM
    value_t *sdram_input_vector = ensemble.parameters.sdram_input_vector;
    spin1_dma_transfer(
        READ_WHOLE_VECTOR, sdram_input_vector, ensemble.input, DMA_READ,
        sizeof(value_t) * ensemble.parameters.n_dims);
}

//! \brief Reads whole input vector from SDRAM
//! \param[in] signal: the index for the next learnt vector to read in
static inline void read_whole_learned_vector(uint32_t signal) {
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
        READ_SPIKE_VECTOR, sdram_spike_vector, ensemble.spikes, DMA_READ,                                  // Direction
        sizeof(uint32_t) * ensemble.sdram_spikes_length);
}

//! \brief cycles vectors
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_read_whole_learnt_vector(uint transfer_id, uint tag){
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
    use(transfer_id);
    use(tag);

    // Process the neurons, then copy the spike vector into SDRAM.
    simulate_neurons(&ensemble, ensemble.spikes);

    // Schedule writing out the spike vector
    spin1_dma_transfer(
        WRITE_SPIKE_VECTOR,         // Tag
        sdram_spikes_vector_local,  // SDRAM address
        ensemble.spikes,            // DTCM addess
        DMA_WRITE,                  // Direction
        spikes_write_size);
}

//! \brief writes next filtered learnt vector or filtered learnt vector
//! \param[in] tag: the tag for the dma
//! \param[in] transfer_id: ??????
void dma_complete_write_filtered_learnt_vector(uint transfer_id, uint tag){
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
void timer_tick(uint timer_count, uint unused) {
    use(timer_count);
    use(unused);
    time++;

    log_debug("Timer tick %u", time);

    // If a fixed number of simulation ticks are specified and these have
    // passed
    if (infinite_run != TRUE && time >= simulation_ticks) {

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

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
    input_filtering_step(&input_filters);
    input_filtering_step(&inhibition_filters);
    input_filtering_step_no_accumulate(&modulatory_filters);
    input_filtering_step_no_accumulate(&learnt_encoder_filters);

    // If there are multiple cores for this ensemble then schedule copying the
    // input vector into SDRAM.  Otherwise if this is address is NULL start
    // processing the neurons.
    if (ensemble.parameters.n_populations > 1) {
        // If there are any learnt input signals to transfer, start transfer of 1st signal
        if(ensemble.parameters.n_learnt_input_signals > 0) {
            write_filtered_learnt_vector(INITIAL_LEARNT_VECTOR_INDEX);
        }
        else { // Otherwise, transfer our section of filtered vector
            write_filtered_vector();
        }
    }
    else {
        // Process the neurons, writing the spikes out into DTCM rather than a
        // shared SDRAM vector.
        simulate_neurons(&ensemble, ensemble.spikes);

        // Apply PES learning to spike vector
        pes_apply(&ensemble, &modulatory_filters);

        // Decode and transmit output
        decode_output_and_transmit(&ensemble);
    }
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
    spin1_memcpy(params, &region_address[START_ENSEMBLE_PARAMS], address),
                 sizeof(ensemble_parameters_t));

    // update the semaphore sdram addresses
    sema_input = params->semaphore_base_address;
    sema_spikes = params->semaphore_base_address + SEMAPHORE_SPIKE_OFFSET;


    // Allocate array to hold pointers to SDRAM learnt input vectors
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
        log_error("cannot allocate dtcm for the local learnt input signals");
        return false;
    }

    // copy in pointers for learnt input vector sdram (local and global)
    uint learnt_vector_pointer = START_LEARNT_INPUT_SIGNALS;
    uint learnt_vector_local_pointer = START_LEARNT_INPUT_SIGNALS + 1;
    for (uint32_t learnt_input_signal=0;
            learnt_input_signal < params->n_learnt_input_signals;
            learnt_input_signal ++) {
        sdram_learnt_input_vector_addresses[learnt_input_signal] =
            region_address[learnt_vector_pointer];
        sdram_learnt_input_vector_local[learnt_vector_local_pointer];
        learnt_vector_pointer += SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR;
        learnt_vector_local_pointer += SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR;
    }

    // process ensemble input
    ensemble.input = spin1_malloc(sizeof(value_t) * params->n_dims);
    if (ensemble.input == NULL){
        log_error("cannot allocate dtcm for the ensemble input");
        return false;
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
        log_error("cannot allocate dtcm for the ensemble learnt input local");
        return false;
    }

    // Loop through learnt input signals and read in as needed
    for(uint32_t input_signal = 0;
            input_signal < params->n_learnt_input_signals; input_signal++) {
        ensemble.learnt_input[i] =
            spin1_malloc(sizeof(value_t) * params->n_dims);
        if (ensemble.learnt_input[i] == NULL){
            log_error(
                "failed to allocate dtcm for learnt input %d", input_signal);
            retrun false;
        }

        // Store local offset
        ensemble.learnt_input_local[input_signal] =
            &ensemble.learnt_input[input_signal][params->input_subspace.offset];

        log_debug(
            "Learnt input signal %u: learnt_input:%08x, "
            "learnt_input_local:%08x, sdram_learnt_input:%08x, "
            "sdram_learnt_input_local:%08x\n", input_signal,
            ensemble.learnt_input[input_signal],
            ensemble.learnt_input_local[input_signal],
            sdram_learnt_input_vector_addresses[input_signal],
            sdram_learnt_input_vector_local[input_signal]);
    }
    return true;
}

//! \brief sets up the spike write size
void set_spike_write_size() {
    // Compute the spike size for writing into SDRAM
    spikes_write_size = params->n_neurons / BITS_IN_WORD;
    if (params->n_neurons % BITS_IN_WORD) {
        spikes_write_size++;
    }
    spikes_write_size *= sizeof(uint32_t);
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
    if (!simulation_initialise(
        data_specification_get_region(SYSTEM, address),
        APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
        &infinite_run, SDP, DMA)) {
        return false;
    }

    // get the ensemble params from sdram
    if (!ensemble_param_read(
            data_specification_get_region(ENSEMBLE_PARAMS, address))){
        return false;
    }

    // set up the spikes write size
    set_spike_write_size();

    // set up input filters and routes for the different filters
    if (!ensemble_setup_input_filteres_and_routes())









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
    spin1_callback_on(MCPL_PACKET_RECEIVED, multicast_packet_payload, MCPL);
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