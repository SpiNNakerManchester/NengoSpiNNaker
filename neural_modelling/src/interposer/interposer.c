/* Parallel implementation of the filter operator
 * ----------------------------------------------
 *
 * Each core in the parallel filter is responsible for receiving packets for a
 * subspace of the value represented by the filter and for transmitting
 * transformed packets for a different subspace.  The general mode of operation
 * on the timer update is:
 *
 *  1. Filter received values
 *  2. Apply local portion of the transform and transmit transformed packets
 *
 * ---
 *
 * The following SDRAM regions are expected:
 *
 *  1. System region (see `filter_parameters_t`)
 *  2. Output keys
 *  3. Filter parameters
 *  4. Filter routes
 *  5. Transform
 *
 * ---
 */


#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <common/nengo_typedefs.h>
#include <common/fixed_point.h>
#include <common/input_filtering.h>
#include <common/packet_queue.h>

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

//! the number of phases to offset the timer by
#define N_PHASES 2

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, SLICE_DATA, KEYS, INPUT_FILTERS, INPUT_ROUTING, TRANSFORM,
    MC_TRANSMISSION
} regions;

//! callback priorities
typedef enum callback_priorities{
    SDP = 0, TIMER = 1, DMA = 2, MCPL = -1, USER=1
} callback_priorities;

//! enum mapping for the mc transmission region
typedef enum mc_transmission_region{
    RANDOM_BACK_OFF, TIME_BETWEEN_SPIKES
}mc_transmission_region;

//! enum mapping for the slice data region
typedef enum slice_data_items{
    N_ATOMS, LO_ATOM, OUTPUT_N_ATOMS
}slice_data_items;

//! enum mapping for the key data
typedef enum key_data_items{
    N_KEYS, START_OF_KEYS
}key_data_items;

// Global variables
// General parameters (system region)
typedef struct filter_parameters
{
    uint32_t input_size;        // Number of columns
    uint32_t input_offset;      // Offset input subspace
    uint32_t output_size;       // Number of rows
} filter_parameters_t;

//! global params for the interposer
static filter_parameters_t params;

//! Locally applied filters
static if_collection_t filters;

//! Transform matrix
static value_t *transform;

//! Multicast keys
uint32_t *keys;

//! Queued multicast packets
static packet_queue_t packets;

//! Indicate if the queue is being handled
static bool queue_processing;

//! provenance data
static unsigned int queue_overflows;

//! \brief runs any functions needed at resume time.
//! \return None
void resume_callback() {
    queue_overflows = 0;
}

//! \brief Multicast packet handling
//! \param[in] payload: the mc packet payload
//! \param[in] key: the mc packet key
void multicast_packet_payload(uint key, uint payload)
{
    // Queue the packet for later processing, if no processing is scheduled then
    // trigger the queue processor.
    if (packet_queue_push(&packets, key, payload))
    {
        if (!queue_processing)
        {
            spin1_trigger_user_event(0, 0);
            queue_processing = true;
        }
    }
    else
    {
        // The packet couldn't be included in the queue, thus it was essentially
        // dropped.
        queue_overflows++;
    }
}

//! \brief takes packets recieved so far and applies input filters
void process_queue()
{
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

            input_filtering_input_with_dimension_offset(
                &filters, key, payload,
                params.input_offset,   // Offset for all packets
                params.input_size - 1  // Max expected dimension
            );
        }
        else
        {
            log_error("Popped packet from empty queue.\n");
            rt_error(RTE_ABORT);
        }
    }
    queue_processing = false;
}

//! \brief user event, fired at timer tick multi-cast reception
void user_event(uint arg0, uint arg1)
{
    use(arg0);
    use(arg1);

    process_queue();
}

//! \brief Timer interrupt callback
//! \param[in] timer_count the number of times this call back has been
//!            executed since start of simulation
//! \param[in] unused for consistency sake of the API always returning two
//!            parameters, this parameter has no semantics currently and thus
//!            is set to 0
//! \return None
void timer_callback(uint timer_count, uint unused) {
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

    // Process any remaining unprocessed packets
    process_queue();

    // Update the filters
    input_filtering_step(&filters);

    // Sleep for a random time so that packet transmission occurs some time
    // after the timer tick.  For shorter simulations this will hide the effect
    // of clock drift for a short period.
    spin1_delay_us(random_backoff_us);

    // Set the next expected time to wait for between spike sending
    expected_time = tc[T1_COUNT] - time_between_spikes;

    // Perform the matrix multiply, transmitting each output value as it is
    // computed.
    for (unsigned int i = 0; i < params.output_size; i++)
    {
        // Get the desired row of the matrix
        value_t *row = transform + i*params.input_size;

        // Get the output key
        uint32_t key = keys[i];

        // Perform the dot-product and transmit the packet
        value_t output = dot_product(params.input_size, row, filters.output);

        // Wait until the expected time to send
        while (tc[T1_COUNT] > expected_time) {
            // Do Nothing
        }
        expected_time -= time_between_spikes;

        // try sending mc packet
        while (!spin1_send_mc_packet(key, bitsk(output), WITH_PAYLOAD))
        {
            spin1_delay_us(1);
        }
    }
}

//! \brief read sdram for mc transmission params
//! \param[in] dsg_address: the sdram address for the mc transmission data
//! \return bool indicating if successful
static bool get_mc_transmission_data(address_t dsg_address){
    random_backoff_us = dsg_address[RANDOM_BACK_OFF];
    time_between_spikes = dsg_address[TIME_BETWEEN_SPIKES];
    return true;
}

//! \brief read sdram for slice params
//! \param[in] dsg_address: the sdram address for the slice data
//! \return bool indicating if successful
static bool get_slice_data(address_t dsg_address){
    params.input_size = dsg_address[N_ATOMS];
    params.input_offset = dsg_address[LO_ATOM];
    params.output_size = dsg_address[OUTPUT_N_ATOMS];
    return true;
}

//! \brief read sdram for key params
//! \param[in] dsg_address: the sdram address for the key data
//! \return bool indicating if successful
static bool get_key_data(address_t dsg_address){
    uint n_keys = dsg_address[N_KEYS];
    if (n_keys > 0){
        keys = spin1_malloc(n_keys * sizeof(uint32_t));
        if (keys == NULL){
            log_error("cannot allocate dtcm for keys");
            return false;
        }
        spin1_memcpy(
            keys, &dsg_address[START_OF_KEYS], n_keys * sizeof(uint32_t));
    }
    return true;
}

//! \brief read sdram for input filters and routes
//! \param[in] address: the sdram address for the dsg region data
//! \return bool indicating if successful
static bool sort_out_input_filters_routes(address_t address){
    uint32_t *words_read;
    words_read = 0;
    if(!input_filtering_initialise_filters(
            &filters, data_specification_get_region(INPUT_FILTERS, address),
            NULL, words_read)){
        return false;
    }
    if(!input_filtering_initialise_routes(
            &filters, data_specification_get_region(INPUT_ROUTING, address),
            words_read)){
        return false;
    }
    if(!input_filtering_initialise_output(&filters, params.input_size)){
        return false;
    }
    return true;
}

//! \brief read sdram for transform params
//! \param[in] dsg_address: the sdram address for the transform data
//! \return bool indicating if successful
static bool get_transform_data(address_t dsg_address){
    uint matrix_size = params.input_size * params.output_size * sizeof(value_t);
    transform = spin1_malloc(matrix_size);
    if (transform == NULL){
        log_error("cannot allocate dtcm for transform data");
        return false;
    }
    spin1_memcpy(transform, dsg_address, matrix_size);
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
    if (!simulation_initialise(
        data_specification_get_region(SYSTEM, address),
        APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
        &infinite_run, SDP, DMA)) {
        return false;
    }

    //! get mc transmission params
    if(!get_mc_transmission_data(
            data_specification_get_region(MC_TRANSMISSION, address))){
        return false;
    }

    //! get slice data
    if(!get_slice_data(
            data_specification_get_region(SLICE_DATA, address))){
        return false;
    }

    //! get key data
    if(!get_key_data(
            data_specification_get_region(KEYS, address))){
        return false;
    }

    //! sort out input filters and routes
    if(!sort_out_input_filters_routes(address)){
        return false;
    }

    //! sort out transform
    if(!get_transform_data(
            data_specification_get_region(TRANSFORM, address))){
        return false;
    }

    // Multicast packet queue
    queue_processing = false;
    packet_queue_init(&packets);
    queue_overflows = 0;

    return true;
}

void c_main(void){
    // Load DTCM data
    uint32_t timer_period;
    if (!initialize(&timer_period)) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick and timer offset (in microseconds)
    spin1_set_timer_tick_and_phase(timer_period, timer_period / N_PHASES);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    spin1_callback_on(MCPL_PACKET_RECEIVED, multicast_packet_payload, MCPL);
    spin1_callback_on(USER_EVENT, user_event, USER);

    // set up run
    simulation_run();

}