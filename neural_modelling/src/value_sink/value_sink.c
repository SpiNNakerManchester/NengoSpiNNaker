#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <recording.h>
#include <common/nengo_typedefs.h>
#include <common/input_filtering.h>
#include <common/packet_queue.h>

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//! The recording flags
static uint32_t recording_flags = 0;

//! the int that represents the bool for if the run is infinite or not.
static uint32_t infinite_run;

//! the time interval parameter CANT USE TIMER_COUNT as not stable
static uint32_t time;

//! the number of atoms which are expected to transmit to this core
static uint32_t input_atoms;

//! The low atom for the atoms expected to transmit to this core
static uint32_t input_lo_atom;

//! collection of filters
if_collection_t filters;

//! Queued multicast packets
static packet_queue_t packets;

//!  Indicate if the queue is being handled
static bool queue_processing = false;

//! queue overload trackers
static unsigned int queue_overflows = 0;

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, SLICE_DATA, FILTERS, FILTER_ROUTING, RECORDING, PROVENANCE_REGION
} regions;

//! callback priorities
typedef enum callback_priorities{
    SDP = 0, TIMER = 2, DMA = 1, USER = 2, MCPL = -1
} callback_priorities;

//! slice_data_region parameters map
typedef enum slice_data_parameters{
    N_ATOMS, LO_ATOM
} slice_data_parameters;

//! enum mapping of extra provenance data items
typedef enum extra_provenance_data_region_entries{
    NUMBER_OF_QUEUE_OVERFLOWS = 0
} extra_provenance_data_region_entries;

//! the recording region channel
#define RECORDING_CHANNEL 0

//! \brief callback for storing extra provenance data items
void c_main_store_provenance_data(address_t provenance_region){
    log_debug("writing other provenance data");

    // store the data into the provenance data region
    provenance_region[NUMBER_OF_QUEUE_OVERFLOWS] = queue_overflows;
    log_debug("finished other provenance data");
}

//! \brief processes multicast packet with payload
//! \param[in] key: the pakcet key
//! \param[in] payload: the packet payload
void mcpl_callback(uint key, uint payload)
{
    // Queue the packet for later processing, if no processing is scheduled then
    // trigger the queue processor.
    log_info("data key %d payload %d converted payload %k",
             key, payload, kbits(payload));
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

//! \brief process the packet queue
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
                input_lo_atom,   // Offset for all packets
                input_atoms - 1  // Max expected dimension
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

//! \brief user event which processes the queue
//! \param [in] arg0: unused value 1
//! \param [in] arg1: unused value 2
void user_event(uint arg0, uint arg1)
{
    use(arg0);
    use(arg1);
    process_queue();
}

//! \brief function for handling resume functionality
void resume_callback(){
    // report error if needed
    //TODO tie into provenance system
    if (queue_overflows > 0){
        log_error("overloaded queue %d times.", queue_overflows);
    }

    recording_reset();

    // reset tracker of queue overflows
    queue_overflows = 0;
}

//! \brief Timer interrupt callback
//! \param[in] timer_count the number of times this call back has been
//!            executed since start of simulation
//! \param[in] unused for consistency sake of the API always returning two
//!            parameters, this parameter has no semantics currently and thus
//!            is set to 0
//! \return None
void timer_callback(uint timer_count, uint unused){
    use(timer_count);
    use(unused);
    time++;

    log_debug("Timer tick %u", time);

    // If a fixed number of simulation ticks are specified and these have
    // passed
    if (infinite_run != TRUE && time >= simulation_ticks) {

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

        // finish recording
        recording_finalise();

        // Subtract 1 from the time so this tick gets done again on the next
        // run
        time -= 1;
        simulation_ready_to_read();
        return;
    }

    // Process any remaining unprocessed packets
    process_queue();

    // Filter inputs, write the latest value to SRAM
    input_filtering_step(&filters);
    recording_record_and_notify(
        RECORDING_CHANNEL,  filters.output, input_atoms * sizeof(value_t),
        NULL);
    recording_do_timestep_update(time);
}

//! \brief entry method for reading the slice data region
//! \param[in] address the absolute SDRAM memory address to which the
//!          slice data starts.
//! \return a boolean which is True if the parameters were read successfully
//!           or False otherwise
static bool read_slice_data_region(address_t address){
    input_atoms = address[N_ATOMS];
    input_lo_atom = address[LO_ATOM];
    return true;
}

//! \brief entry method for reading the filter data region
//! \param[in] dsg_address the absolute SDRAM memory address to which the
//!          dsg regions map data starts.
//! \return a boolean which is True if the parameters were read successfully
//!           or False otherwise
static bool set_up_filters(address_t dsg_address){
    if(!input_filtering_initialise_output(&filters, input_atoms)){
        return false;
    }

    uint32_t *words_read;
    words_read = 0;

    if(!input_filtering_initialise_filters(
            &filters, data_specification_get_region(FILTERS, dsg_address),
            NULL, words_read)){
        return false;
    }
    if(!input_filtering_initialise_routes(
            &filters,
            data_specification_get_region(FILTER_ROUTING, dsg_address),
            words_read)){
        return false;
    }
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

    // get slice data
    if (!read_slice_data_region(
            data_specification_get_region(SLICE_DATA, address))){
        return false;
    }

    // set up filters
    if (!set_up_filters(address)){
        return false;
    }

    //! set up recording
    if (!recording_initialize(
            data_specification_get_region(RECORDING, address),
            &recording_flags)){
        log_error("failed to setup recording");
        return false;
    }
    log_debug("Recording flags = 0x%08x", recording_flags);

    // Multicast packet queue
    packet_queue_init(&packets);

    // sort out provenance region
    simulation_set_provenance_data_address(
        data_specification_get_region(PROVENANCE_REGION, address));

    // set up provenance function
    simulation_set_provenance_function(
        c_main_store_provenance_data,
        data_specification_get_region(PROVENANCE_REGION, address));

    // passed
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

    // Register callbacks
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    spin1_callback_on(MCPL_PACKET_RECEIVED,  mcpl_callback, MCPL);
    spin1_callback_on(USER_EVENT, user_event, USER);

    // set up run
    simulation_run();
}
