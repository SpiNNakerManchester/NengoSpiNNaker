#ifndef NEURAL_MODELLING_VOJA_H
#define NEURAL_MODELLING_VOJA_H

#include <common/input_filtering.h>
#include "common-typedefs.h"

#define NOT_EFFECTED_BY_ERROR_RATE -1

// Structure containing parameters and state required for Voja learning
typedef struct voja_parameters_t
{
    // Scalar learning rate used in Voja encoder delta calculation
    value_t learning_rate;

    // Index of the input signal filter that contains
    // learning signal. -1 if there is no learning signal
    int32_t learning_signal_filter_index;

    // Offset into encoder to apply Voja
    uint32_t encoder_offset;

    // Index of the input signal filter than contains
    // the decoded input from the pre-synaptic ensemble
    uint32_t decoded_input_filter_index;
} voja_parameters_t;

#define CONSTANT_ERROR_RATE 1.0k

//----------------------------------
// External variables
//----------------------------------
//! number of voja learning rules
extern uint32_t g_num_voja_learning_rules;

//! set of voja learning rules parameters
extern voja_parameters_t *g_voja_learning_rules;

//! param 1/radius
extern value_t g_voja_one_over_radius;

//----------------------------------
// Inline functions
//----------------------------------
//! \brief Helper to get the Voja learning rate - can be modified at runtime
//!        with a signal
//! \param[in] modulatory_filters: the filters for modulatory filters
//! \param[in] parameters: the voja learning rules parameters
//! \return the learning rate
static inline value_t voja_get_learning_rate(
        const voja_parameters_t *parameters,
        const if_collection_t *modulatory_filters){
    // If a learning signal filter index is specified, read the value
    // from it's first dimension and multiply by the constant error rate
    if(parameters->learning_signal_filter_index != NOT_EFFECTED_BY_ERROR_RATE){
        const if_filter_t *decoded_learning_input =
            &modulatory_filters->filters[
                parameters->learning_signal_filter_index];
        value_t positive_learning_rate =
            CONSTANT_ERROR_RATE + decoded_learning_input->output[0];
        return parameters->learning_rate * positive_learning_rate;
    }
        // Otherwise, just return the constant learning rate
    else{
        return parameters->learning_rate;
    }
}

//! \brief When using non-filtered activity, applies Voja when neuron spikes
//! \param[in] encoder_vector:
//! \param[in] gain:
//! \param[in] learnt_input:
//! \param[in] modulatory_filters:
//! \param[in] n_dims:
static inline void voja_neuron_spiked(
        value_t *encoder_vector, value_t gain, uint32_t n_dims,
        const if_collection_t *modulatory_filters,
        const value_t **learnt_input){
    // Loop through all the learning rules
    for(uint32_t learning_rule = 0; learning_rule < g_num_voja_learning_rules;
        learning_rule++){

        const voja_parameters_t *parameters =
            &g_voja_learning_rules[learning_rule];
        // Get learning rate
        const value_t learning_rate =
            voja_get_learning_rate(parameters, modulatory_filters);

        // Get correct signal from learnt input
        const value_t *decoded_input_signal =
            learnt_input[parameters->decoded_input_filter_index];

        // Get this neuron's encoder vector, offset by the encoder offset
        value_t *learnt_encoder_vector =
            encoder_vector + parameters->encoder_offset;

        // Calculate scaling factor for input
        const value_t input_scale =
            learning_rate * gain * g_voja_one_over_radius;

        // Loop through input dimensions
        for(uint dimension = 0; dimension < n_dims; dimension++){
            learnt_encoder_vector[dimension] +=
                (input_scale * decoded_input_signal[dimension]) -
                (learning_rate * learnt_encoder_vector[dimension]);
        }
    }
}

//! \brief Copy in data controlling the Voja learning
//!        rule from the Voja region of the Ensemble.
//! \param[in] address: the dsg address for the voja region
bool voja_initialise(address_t address);

#endif //NEURAL_MODELLING_VOJA_H
