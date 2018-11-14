/*! \file
 *
 *  \brief utility class which ensures that format of bools being recorded is
 *   done in a standard way
 *
 *
 *  \details The API includes:
 *     - out_bools_reset
 *          clears the memory used as a tracker for the next set of bools
 *          which will be recorded to SDRAM at some point
 *     - out_bools_initialize
 *          initialises a piece of memory which can contain a flag to say if
 *          any source has bool set between resets
 *     - out_bools_record
 *          records the current set of flags for each bool source into the
 *          bool recording region in SDRAM (flags to deduce which regions are
 *           active are handed to this method due to recording not containing
 *           them itself). TODO change the recording.h and recording.c to
 *           contain the channels itself.
 *     - out_bools_is_empty
 *          helper method which checks if the current bools flags have any
 *          recorded for use.
 *     - out_bools_is_bool
 *          helper method which checks if a given source has bool set since the
 *           last reset.
 *     - out_bools_print
 *          a debug function that when the model is compiled in DEBUG mode will
            record into SDRAM the bools that are currently been recorded as
            having bool set since the last reset command
 *     - out_bools_set_bool
 *          helper method which allows models to state that a given bool source
            has bool set since the last reset.
 */

#ifndef _OUT_SPIKES_H_
#define _OUT_SPIKES_H_

#include <bit_field.h>
#include <recording.h>

extern bit_field_t out_bools;

//! \brief clears the currently recorded bools
void out_bools_reset();

//! \brief initialise the recording of bools
//! \param[in] max_bool_sources the number of bool sources to be recorded
//! \return True if the initialisation was successful, false otherwise
bool out_bools_initialize(size_t max_bool_sources);

//! \brief flush the recorded bools - must be called to do the actual
//!        recording
//! \param[in] channel The channel to record to
//! \param[in] time The time at which the recording is being made
//! \param[in] n_words The number of words of the buffer to record - allows
//!                    the buffer to be allocated larger than needed
//! \param[in] callback Callback to call when the recording is done
//                      (can be NULL)
bool out_bools_record(
    uint8_t channel, uint32_t time, uint32_t n_words,
    recording_complete_callback_t callback);

//! \brief Check if any bools have been recorded
//! \return True if no bools have been recorded, false otherwise
bool out_bools_is_empty();

//! \brief Check if a given neuron has been recorded to bool
//! \param[in] bool_source_index The index of the neuron.
//! \return true if the bool source has been recorded to bool
bool out_bools_is_bool(index_t bool_source_index);

//! \brief print out the contents of the output bools (in DEBUG only)
void out_bools_print();

//! \brief Indicates that a neuron has bool set
//! \param[in] bool_source_index The index of the neuron that has bool set
static inline void out_bools_set_bool(index_t bool_source_index) {
    bit_field_set(out_bools, bool_source_index);
}

#endif // _OUT_SPIKES_H_
