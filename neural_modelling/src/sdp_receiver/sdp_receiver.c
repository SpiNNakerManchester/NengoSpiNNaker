#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <common/nengo_typedefs.h>


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

//! start value for
#define START_VALUE_OUTPUT 0x00000000

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, SDP_PORT, KEYS, MC_TRANSMISSION_PARAMS, PROVENANCE_REGION
} regions;

//! enum mapping sdp_port region
typedef enum sdp_port_region {
    SDP_PORT_VALUE
} sdp_port_region;

//! enum mapping for the mc transmission region
typedef enum mc_transmission_region{
    RANDOM_BACK_OFF, TIME_BETWEEN_SPIKES
}mc_transmission_region;

//! enum mapping for keys region
typedef enum key_region{
    N_KEYS, START_OF_KEYS
} key_region;

//! callback priorities
typedef enum callback_priorities{
    SDP = -1, TIMER = 1, DMA = 0
} callback_priorities;


/** \brief Shared Rx parameters.
 */
typedef struct sdp_rx_parameters {
    uint n_dimensions;        //!< Number of dimensions represented
    uint current_dimension;   //!< Index of the currently selected dimension

    value_t *output;          //!< Currently cached output value
    bool *fresh;              //!< Freshness of output
    uint *keys;               //!< Output keys
} sdp_rx_parameters_t;

sdp_rx_parameters_t g_sdp_rx; //!< Global parameters

//! \brief function for handling resume functionality
void resume_callback(){}

//! \brief Receive packed data packed in SDP message
//! \param[in] mailbox: the box
//! \param[in] port: the sdp port it was received from
void sdp_received(uint mailbox, uint port) {
    use(port);
    log_info("received sdp packet");
    sdp_msg_t *message = (sdp_msg_t*) mailbox;

    // Copy the data into the output buffer
    // Mark values as being fresh
    value_t * data = (value_t*) message->data;
    for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
        g_sdp_rx.output[d] = data[d];
        g_sdp_rx.fresh[d] = true;
    }
    spin1_msg_free(message);
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

    // Sleep for a random time so that packet transmission occurs some time
    // after the timer tick.  For shorter simulations this will hide the effect
    // of clock drift for a short period.
    spin1_delay_us(random_backoff_us);

    // Set the next expected time to wait for between spike sending
    expected_time = tc[T1_COUNT] - time_between_spikes;

    // transmit message for each neuron
    for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
        if (g_sdp_rx.fresh[d]) {
            // Wait until the expected time to send
            while (tc[T1_COUNT] > expected_time) {
                // Do Nothing
            }
            expected_time -= time_between_spikes;

            // try to send message
            while(!spin1_send_mc_packet(
                    g_sdp_rx.keys[d], bitsk(g_sdp_rx.output[d]),
                    WITH_PAYLOAD)){
                spin1_delay_us(1);
            }

            // no longer fresh
            g_sdp_rx.fresh[d] = false;
        }
    }
}

//! \brief read sdram for the sdp port data
//! \param[in] dsg_address: the sdram address where this data is
//! \param[out] sdp_port: the sdp port to listen on
//! \return bool indicating if successful
static bool get_sdp_port_data(address_t dsg_address, uint32_t *sdp_port){
    *sdp_port = dsg_address[SDP_PORT_VALUE];
    return true;
}

// \brief read sdram for the key data
//! \param[in] dsg_address: the sdram address for the key data
//! \return bool indicating if successful
static bool get_keys_data(address_t dsg_address){
    g_sdp_rx.n_dimensions = dsg_address[N_KEYS];
    g_sdp_rx.keys = spin1_malloc(g_sdp_rx.n_dimensions * sizeof(uint));
    if (g_sdp_rx.keys == NULL){
        log_error("failed to allocate dtcm for the keys");
        return false;
    }
    spin1_memcpy(
        g_sdp_rx.keys, &dsg_address[START_OF_KEYS],
        g_sdp_rx.n_dimensions * sizeof(uint));
    return true;
}

//! \brief read sdram for mc transmission params
//! \param[in] dsg_address: the sdram address for the mc transmission data
//! \return bool indicating if successful
static bool get_mc_transmission_data(address_t dsg_address){
    random_backoff_us = dsg_address[RANDOM_BACK_OFF];
    time_between_spikes = dsg_address[TIME_BETWEEN_SPIKES];
    return true;
}

//! \brief allocates DTCM for some params and instantiates them to correct values
//! \return bool indicating if successful
static bool malloc_dtcm_and_init(){
    // allocate dtcms
    g_sdp_rx.output = spin1_malloc(g_sdp_rx.n_dimensions * sizeof(value_t));
    if (g_sdp_rx.output == NULL){
        log_error("failed to allocate dtcm for output");
        return false;
    }
    g_sdp_rx.fresh = spin1_malloc(g_sdp_rx.n_dimensions * sizeof(bool));
    if (g_sdp_rx.fresh == NULL){
        log_error("failed to allocate dtcm for fresh");
        return false;
    }

    // set the allocated dtcm to correct values
    for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
        g_sdp_rx.output[d] = START_VALUE_OUTPUT;
        g_sdp_rx.fresh[d] = false;
    }
    return true;
}

//! Initialises the model by reading in the regions and checking recording
//! data.
//! \param[out] timer_period a pointer for the memory address where the timer
//!            period should be stored during the function.
//! \param[out] sdp_port The SDP port on which to listen for packets
//! \return boolean of True if it successfully read all the regions and set up
//!         all its internal data structures. Otherwise returns False
static bool initialize(uint32_t *timer_period, uint32_t *sdp_port){

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

    // get the transmission params for spreading purposes
    if (!get_mc_transmission_data(
        data_specification_get_region(MC_TRANSMISSION_PARAMS, address))){
        return false;
    }

    // get sdp port for sending sdp messages
    if (!get_sdp_port_data(
            data_specification_get_region(SDP_PORT, address), sdp_port)){
        return false;
    }

    // get key data
    if (!get_keys_data(
            data_specification_get_region(KEYS, address))){
        return false;
    }

    // allocate the dtcm and sdram for sdp fresh and output
    if(!malloc_dtcm_and_init()){
        return false;
    }

    // sort out provenance region
    simulation_set_provenance_data_address(
        data_specification_get_region(PROVENANCE_REGION, address));

    //passed
    return true;
}



void c_main(void){
    // Load DTCM data
    uint32_t timer_period;
    uint32_t sdp_port = 0;
    if (!initialize(&timer_period, &sdp_port)) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick (in microseconds)
    spin1_set_timer_tick(timer_period);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    simulation_sdp_callback_on(sdp_port, sdp_received);

    // set up run
    simulation_run();

}