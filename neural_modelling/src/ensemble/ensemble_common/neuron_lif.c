#include "neuron_lif.h"
#include <debug.h>
#include <common/constants.h>

//! initial state for the voltages and refractory states.
#define INITIAL_STATE 0

//! \brief Prepare neuron state
//! \param[in] address: SDRAM address of neuron parameters
//! \param[in] ensemble: Generic ensemble state
//! \param[out] sdram_words_read: the number of words in sdram read
//! \return bool stating if the prepare state succeeded or not
bool lif_prepare_state(
        ensemble_state_t *ensemble, uint32_t *address,
        uint32_t *sdram_words_read) {
    // Get the number of neurons
    uint32_t n_neurons = ensemble->parameters.n_neurons;

    // Prepare space for neuron parameters
    ensemble->state = spin1_malloc(sizeof(lif_states_t));
    if (ensemble->state == NULL){
        log_error("failed to allocate dtcm for the ensemble state");
        return false;
    }
    lif_states_t *state = ensemble->state;

    // Allocate space for voltages (and zero)
    state->voltages = spin1_malloc(sizeof(value_t) * n_neurons);
    if (state->voltages == NULL){
        log_error("failed to allocate dtcm for the ensemble state voltage");
        return false;
    }
    memset(state->voltages, INITIAL_STATE, sizeof(value_t) * n_neurons);

    // Allocate space for refractory counters
    state->refractory = spin1_malloc(sizeof(uint32_t) * n_neurons);
    if(state->refractory == NULL){
        log_error("failed to allocate dtcm for the ensemble state refactory");
        return false;
    }
    memset(state->refractory, INITIAL_STATE, sizeof(uint32_t) * n_neurons);

    // Copy in LIF parameters
    spin1_memcpy(&state->parameters, address, sizeof(lif_parameters_t));
    sdram_words_read = int(sizeof(lif_parameters_t) / BYTES_TO_WORD_CONVERSION);
    return true;
}
