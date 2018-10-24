#ifndef NEURAL_MODELLING_NEURON_LIF_H
#define NEURAL_MODELLING_NEURON_LIF_H

#include "stdfix-full-iso.h"
#include <ensemble/ensemble.h>

//! accum set to 0.0
#define ZERO_ACCUM_CONSTANT 0.0k

//! accum set to 1.0
#define ONE_ACCUM_CONSTANT 1.0k

//! accum set to 2.0
#define TWO_ACCUM_CONSTANT 2.0k

//! tau variables for an ensemble of LIF neurons
typedef struct lif_parameters{
    // Function of neuron time constant
    value_t exp_dt_over_tau_rc;
    // Refractory period
    uint32_t tau_ref;
} lif_parameters_t;

//! lif params and states
typedef struct lif_states{
    // Neuron parameters
    lif_parameters_t parameters;
    // Neuron voltages
    value_t *voltages;
    // Refractory counters
    uint8_t *refractory;
} lif_states_t;

//! \brief Prepare neuron state
//! \param[in] address: SDRAM address of neuron parameters
//! \param[in] ensemble: Generic ensemble state
//! \return bool stating if the state prepare was successful
bool lif_prepare_state(ensemble_state_t *ensemble, uint32_t *address);

//! \brief Get the refractory counter for a given neuron
//! \param[in] neuron: Index of the neuron to simulate
//! \param[in] state:Pointer to neuron state(s)
static inline uint8_t neuron_refractory(
        const uint32_t neuron, const void *state) {
    // Cast the state to LIF state type
    lif_states_t *lif_state = (lif_states_t *) state;

    // Return the refractory state for the given neuron
    return lif_state->refractory[neuron];
}

//! \brief Decrement the refractory counter for a given neuron
//! \param[in] state: Pointer to neuron state(s)
//! \param[in] neuron: Index of the neuron to simulate
static inline void neuron_refractory_decrement(
        const uint32_t neuron, const void *state) {
    // Cast the state to LIF state type
    lif_states_t *lif_state = (lif_states_t *) state;

    // Decrement the refractory state for the given neuron
    lif_state->refractory[neuron]--;
}

//! \brief Perform a single neuron step
//! \param[in] neuron: Index of the neuron to simulate
//! \param[in] state: Pointer to neuron state(s)
//! \param[in] input: Input to the neuron
//! \param[in] rec_voltages: Pointer to voltage recording
static inline bool neuron_step(
        const uint32_t neuron, const value_t input,
        const void *state, recording_buffer_t *rec_voltages) {

    // Cast the state to LIF state type
    lif_states_t *lif_state = (lif_states_t *) state;

    // Compute the change in voltage
    value_t voltage = lif_state->voltages[neuron];
    value_t delta_v =
        (input - voltage) * lif_state->parameters.exp_dt_over_tau_rc;

    // Update the voltage, but clip it to 0.0
    voltage += delta_v;
    if (bitsk(voltage) < bitsk(ZERO_ACCUM_CONSTANT)){
        voltage = ZERO_ACCUM_CONSTANT;
    }

    // If the neuron hasn't fired then simply store the voltage and return false
    // to indicate that no spike was produced.
    if (bitsk(voltage) <= bitsk(ONE_ACCUM_CONSTANT)){
        lif_state->voltages[neuron] = voltage;
        record_voltage(rec_voltages, neuron, voltage);
        return false;
    }

    // The neuron has spiked, so we prepare to set the voltage and refractory
    // period for the next simulation period.
    uint8_t tau_ref = (uint8_t) lif_state->parameters.tau_ref;
    voltage -= ONE_ACCUM_CONSTANT;

    // If the overshoot was particularly big further decrease the neuron voltage
    // and refractory period.
    if (bitsk(voltage) > bitsk(TWO_ACCUM_CONSTANT)){
        tau_ref--;
        voltage -= delta_v;
    }

    // Store the refractory period and voltage, return true to indicate that a
    // spike occurred.
    lif_state->refractory[neuron] = tau_ref;
    lif_state->voltages[neuron] = voltage;
    return true;
}


#endif //NEURAL_MODELLING_NEURON_LIF_H
