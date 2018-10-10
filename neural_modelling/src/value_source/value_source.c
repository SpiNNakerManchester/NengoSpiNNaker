#include <spin1_api.h>
#include <data_specification.h>
#include <debug.h>
#include <simulation.h>
#include <spin1_api.h>
#include "slots.h"
#include "nengo_typedefs.h"

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//! the int that represents the bool for if the run is infinite or not.
static uint32_t infinite_run;

//! the time interval parameter CANT USE TIMER_COUNT as not stable
static uint32_t time;

//! The expected time to wait between spikes
static uint32_t time_between_spikes;

//! The random backoff between timer ticks to desynchronize sources
static uint32_t random_backoff_us;

//!< Number of neurons (frame length)
static uint n_neurons;

//! States if the source should repeat once finished
static bool is_cyclic;

//! number of full blocks
static uint n_blocks;

//! total blocks avilable (full and partial)
static uint total_blocks;

//! length of a FULL block in frames
static uint block_length;

//! length of the last PARTIAL block in frames
static uint partial_block_length;

//! the current block id
uint current_block = 0;

//! n neurons being emulated
static uint n_neurons;

//! flag for when ive finished packets but not finished time steps
bool has_finished = false;

//! holder for the buffers
slots_t slots;

//! array of keys (1 per neuron)
uint* keys;

//! location of blocks in DRAM
value_t* blocks;

//! The expected current clock tick of timer_1
static uint32_t expected_time;

//! how much DTCM is to be allocated for the buffers
#define DTCM_FOR_BUFFERS (20 * 1024)  // 20 KB of DTCM

//! enum mapping region ids to regions in python
typedef enum regions {
    SYSTEM, OUTPUT_REGION, KEY_REGION, NEURON_REGION, RECORDING
} regions;

//! enum mapping neuron params
typedef enum neuron_params {
    IS_CYCLIC, N_NEURONS, RANDOM_BACK_OFF, TIME_BETWEEN_SPIKES
} neuron_params;

//! callback priorities
typedef enum callback_priorities{
    SDP = -1, TIMER = 0, DMA = 1
} callback_priorities;

typedef struct value_source_params {
    uint time_between_spikes;     //!< Time step of the ValueSource in us
    uint n_dims;        //!< Number of output dimensions (frame length)
    uint flags;         //!< Flags
    uint n_blocks;      //!< Number of FULL blocks
    uint block_length;  //!< Length of a FULL block in frames
    uint partial_block; //!< Length of the last PARTIAL block in frames
} value_source_params;


//! \brief runs any functions needed at resume time.
//! \return None
void resume_callback() {

}

//! \brief Sends spikes linked to the given time step
//! \return None
void process_spikes_for_this_time_step(){

    // Transmit a MC packet for each value in the current frame
    for (uint neuron_id = 0; neuron_id < n_neurons; neuron_id++) {

        // Wait until the expected time to send
        while (tc[T1_COUNT] > expected_time) {
            // Do Nothing
        }

        expected_time -= time_between_spikes;

        // send a given spike
        while(!spin1_send_mc_packet(
                keys[neuron_id],
                slots.current->data[
                    slots.current->current_pos*n_neurons + neuron_id],
                WITH_PAYLOAD)) {
            spin1_delay_us(1);
        }
    }
}

//! \brief loads in next buffer
//! \return None
void bring_in_next_block(){
    if (slots.current->current_pos == 0) {
        if (total_blocks > 1) {
            // More than one block, need to copy in subsequent block
            value_t *s_addr = &blocks[
                (current_block + 1) * block_length * n_neurons];

            if (current_block == n_blocks - 1) {
                // Subsequent block is the LAST block
                spin1_dma_transfer(
                    0, s_addr, slots.next->data, DMA_READ,
                    partial_block_length * n_neurons * sizeof(value_t));
                slots.next->length = partial_block_length;
            } else if (current_block == n_blocks) {
                // Current block is the LAST block
                if (is_cyclic) {
                    // We are wrapping, so next block is the FIRST block
                    spin1_dma_transfer(
                        0, blocks, slots.next->data, DMA_READ,
                        block_length * n_neurons * sizeof(value_t));
                    slots.next->length = block_length;
                }
            } else {
                // Nothing special about subsequent block
                spin1_dma_transfer(
                    0, s_addr, slots.next->data, DMA_READ,
                    block_length * n_neurons * sizeof(value_t));
                slots.next->length = block_length;
            }
        }
    }
}

//! \brief switch blocks for cyclic stuff
//! \return None
void switch_block_if_necessary(){
    slots.current->current_pos++;
    if (slots.current->current_pos == slots.current->length) {
        // We've reached the end of the current slot, progress or wrap
        if (n_blocks == 1) {
            // Only one block: wrap or exit
            if (is_cyclic) {
                // Function is periodic: wrap to start
                slots.current->current_pos = 0;
            } else {
                // Function is not periodic: exit
                has_finished = true;
            }
        } else {
            // Multiple blocks: next, wrap or exit
            if (current_block == n_blocks - 1 && !(is_cyclic)) {
                // Last block, aperiodic: exit
                has_finished = true;
            } else {
                // Not last block, or periodic: next
                slots_progress(&slots);
                current_block++;

                // Wrap if necessary
                if (current_block == n_blocks)
                    current_block = 0;
            }
        }
    }
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

    if (!has_finished) {
        // sends said spikes as needed
        process_spikes_for_this_time_step();

        // loads next buffer
        bring_in_next_block();

        // cyclic buffers
        switch_block_if_necessary();
    }
}

//! \brief entry method for reading the neuron params
//! \param[in] address the absolute SDRAM memory address to which the
//!            NEURON_PARAMS starts.
//! \return a boolean which is True if the parameters were read successfully or
//!         False otherwise
static bool read_neuron_region(address_t address){
    is_cyclic = address[IS_CYCLIC];
    n_neurons = address[N_NEURONS];
    random_backoff_us = address[RANDOM_BACK_OFF];
    time_between_spikes = address[TIME_BETWEEN_SPIKES];


    block_length = (int) 20 * 1024 / (n_neurons * 4.0);
    n_blocks = (int) (simulation_ticks / block_length);
    partial_block_length = simulation_ticks % block_length;
    return true;
}

//! \brief entry method for reading the key region
//! \param[in] address the absolute SDRAM memory address to which the
//!          NEURON_PARAMS starts.
//! \return a boolean which is True if the parameters were read successfully
//!           or False otherwise
static bool read_key_region(address_t address){
    // Make space for keys
    keys = spin1_malloc(n_neurons * sizeof(uint));
    if (keys == NULL) {
        return false;
    }
    spin1_memcpy(keys, address, n_neurons * sizeof(uint));
    return true;
}

//! \brief entry method for reading the output data region
//! \param[in] address the absolute SDRAM memory address to which the
//!          output data starts.
//! \return a boolean which is True if the parameters were read successfully
//!           or False otherwise
static bool read_output_region(address_t address){
    total_blocks = n_blocks + (partial_block_length > 0 ? 1 : 0);
    blocks = (value_t *) address;
    slots_progress(&slots);
    if (total_blocks > 1){
        spin1_memcpy(
            slots.current->data, address,
            n_neurons * block_length * sizeof(value_t));
        slots.current->length = block_length;
    }
    else {
        spin1_memcpy(
            slots.current->data, address,
            n_neurons * partial_block_length * sizeof(value_t));
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

    // get param region NEURON_REGION
    if(!read_neuron_region(
            data_specification_get_region(NEURON_REGION, address))) {
        return false;
    }

    // setup slots
    if(!initialise_slots(&slots, DTCM_FOR_BUFFERS)){
        return false;
    }

    // get output region
    if(!read_output_region(
            data_specification_get_region(OUTPUT_REGION, address))){
        return false;
    }

    // get keys region
    if(!read_key_region(
            data_specification_get_region(KEY_REGION, address))){
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

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);

    // set up run
    simulation_run();

}