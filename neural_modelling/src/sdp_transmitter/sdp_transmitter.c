#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <common/nengo_typedefs.h>
#include <common/input_filtering.h>


//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//! the int that represents the bool for if the run is infinite or not.
static uint32_t infinite_run;

//! the time interval parameter CANT USE TIMER_COUNT as not stable
static uint32_t time;

//! the time left to delay for the current setup
uint delay_remaining;

/** \brief Shared Tx parameters.
  */
typedef struct sdp_tx_parameters {
    uint transmission_delay; //!< Number of ticks between output transmissions
    uint n_dimensions;       //!< Number of dimensions to represent
    value_t *input;          //!< Input buffer
    uint tag_id;             //!< TAG id for transmissions
    uint dest_addr;          //!< address to send to
} sdp_tx_parameters_t;

//! parameter holder
sdp_tx_parameters_t g_sdp_tx;



//! input queue.
if_collection_t g_input;

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, TRANSMITTER, FILTERS, FILTER_ROUTING
} regions;

//! enum mapping transmitter
typedef enum transmitter_region_elements {
    SIZE_IN, TRANSMISSION_DELAY, TAG_ID, DEST_ADDR
} transmitter_region_elements;

//! callback priorities
typedef enum callback_priorities{
    SDP = 0, TIMER = 2, MCPL = -1, DMA = 2
} callback_priorities;

//! timeout for trying sdp transmission
#define SDP_TIMEOUT 100

//! flags for a message
#define MESSAGE_FLAGS 0x07

//! \brief function for handling resume functionality
void resume_callback(){}

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

    // Update the filters
    input_filtering_step(&g_input);

    // Increment the counter and transmit if necessary
    delay_remaining--;
    if (delay_remaining == 0) {
        delay_remaining = g_sdp_tx.transmission_delay;

        // Construct and transmit the SDP Message
        sdp_msg_t message;
        message.dest_addr = g_sdp_tx.dest_addr;
        message.dest_port = PORT_ETH;
        message.srce_addr = sv->p2p_addr;  // Sender P2P address
        message.srce_port = spin1_get_id();
        message.flags = MESSAGE_FLAGS;              // No reply expected
        message.tag = g_sdp_tx.tag_id;      // Send to IPtag

        message.cmd_rc = 1;
        spin1_memcpy(
            message.data, g_sdp_tx.input,
            g_sdp_tx.n_dimensions * sizeof(value_t));

        message.length =
            sizeof(sdp_hdr_t) + sizeof(cmd_hdr_t) +
            g_sdp_tx.n_dimensions * sizeof(value_t);

        while (!spin1_send_sdp_msg(&message, SDP_TIMEOUT)) {
            log_error("cant send sdp packet");
        }
    }
}

//! \brief function to handle the reception of a mc packet
//! \param[in] key: the key of the mc packet
//! \param[in] payload: the payload of the mc packet.
void mcpl_callback(uint key, uint payload) {
    input_filtering_input(&g_input, key, payload);
}

//! \brief entry method for reading the transmission data region
//! \param[in] dsg_address the absolute SDRAM memory address to which the
//!          dsg regions map data starts.
//! \return a boolean which is True if the parameters were read successfully
//!           or False otherwise
static bool read_transmission_region(address_t dsg_address){
    g_sdp_tx.n_dimensions = dsg_address[SIZE_IN];
    g_sdp_tx.transmission_delay = dsg_address[TRANSMISSION_DELAY];
    g_sdp_tx.tag_id = dsg_address[TAG_ID];
    g_sdp_tx.dest_addr = dsg_address[DEST_ADDR];
    return true;
}


//! Initialises the model by reading in the regions and checking recording
//! data.
//! \param[out] timer_period a pointer for the memory address where the timer
//!            period should be stored during the function.
//! \param[out] update_sdp_port The SDP port on which to listen for rate
//!             updates
//! \return boolean of True if it successfully read all the regions and set up
//!         all its internal data structures. Otherwise returns False
static bool initialize(uint32_t *timer_period) {

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

    // process transmission region
    if (!read_transmission_region(
            data_specification_get_region(TRANSMITTER, address))){
        return false;
    }

    // build input filtered queue
    if(!input_filtering_initialise_output(&g_input, g_sdp_tx.n_dimensions)){
        return false;
    }
    g_sdp_tx.input = g_input.output;

    // handle filters
    if(!input_filtering_initialise_filters(
            &g_input, data_specification_get_region(FILTERS, address), NULL)){
        return false;
    }

    if(!input_filtering_initialise_routes(
            &g_input, data_specification_get_region(FILTER_ROUTING, address))){
        return false;
    }

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

    // Set timer tick (in microseconds)
    spin1_set_timer_tick(timer_period);
    spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, MCPL);
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);

    // set up run
    simulation_run();
}