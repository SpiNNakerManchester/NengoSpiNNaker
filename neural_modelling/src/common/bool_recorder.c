/*! \file
 *
 *  \brief the implementation of the bool_recorder.h interface.
 */

#include "bool_recorder.h"

#include <recording.h>
#include <debug.h>

// Globals
typedef struct timed_out_bools{
    uint32_t time;
    uint32_t out_bools[];
} timed_out_bools;

static timed_out_bools *bools;
bit_field_t out_bools;
static size_t out_bools_size;


//! \brief clears the currently recorded bools
void out_bools_reset() {
    clear_bit_field(out_bools, out_bools_size);
}

//! \brief initialise the recording of bools
//! \param[in] max_bool_sources the number of bool sources to be recorded
//! \return True if the initialisation was successful, false otherwise
bool out_bools_initialize(size_t max_bool_sources) {
    out_bools_size = get_bit_field_size(max_bool_sources);
    log_debug("Out bool size is %u words, allowing %u bool sources",
              out_bools_size, max_bool_sources);
    bools = (timed_out_bools *) spin1_malloc(
        sizeof(timed_out_bools) + (out_bools_size * sizeof(uint32_t)));
    if (bools == NULL) {
        log_error("Out of DTCM when allocating out_bools");
        return false;
    }
    out_bools = &(bools->out_bools[0]);
    out_bools_reset();
    return true;
}

bool out_bools_record(
        uint8_t channel, uint32_t time, uint32_t n_words,
        recording_complete_callback_t callback) {
    if (out_bools_is_empty()) {
        log_info("attempt calling callback");
        if (callback != NULL) {
            log_info("calling callback");
            callback();
         }
        return false;
    } else {
        bools->time = time;
        return recording_record_and_notify(
            channel, bools, (n_words + 1) * sizeof(uint32_t),
            callback);
    }
}

//! \brief Check if any bools have been recorded
//! \return True if no bools have been recorded, false otherwise
bool out_bools_is_empty() {
    return (empty_bit_field(out_bools, out_bools_size));

}

//! \brief Check if a given neuron has been recorded to bool
//! \param[in] bool_source_index The index of the neuron.
//! \return true if the bool source has been recorded to bool
bool out_bools_is_bool(index_t neuron_index) {
    return (bit_field_test(out_bools, neuron_index));
}

//! \brief print out the contents of the output bools (in DEBUG only)
void out_bools_print() {
#if LOG_LEVEL >= LOG_DEBUG
    log_debug("out_bools:\n");

    if (!out_bools_is_empty()) {
        log_debug("-----------\n");
        print_bit_field(out_bools, out_bools_size);
        log_debug("-----------\n");
    }
#endif // LOG_LEVEL >= LOG_DEBUG
}

void out_bool_info_print(){
    log_debug("-----------\n");
    index_t i; //!< For indexing through the bit field

    for (i = 0; i < out_bools_size; i++) {
        log_debug("%08x\n", out_bools[i]);
    }
    log_debug("-----------\n");
}
