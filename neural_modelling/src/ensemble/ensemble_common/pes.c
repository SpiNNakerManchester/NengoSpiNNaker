#include "pes.h"
#include <debug.h>

//! Structure containing parameters and state required for PES learning
typedef struct pes_parameters_t
{
    // Scalar learning rate used in PES decoder delta calculation
    value_t learning_rate;

    // Index of the modulatory input signal filter that contains error signal
    uint32_t error_sig_index;

    // Start and end dimensions of the error signal to use in this vertex
    uint32_t error_start_dim;
    uint32_t error_end_dim;

    // Which row decoder to apply PES learning from
    uint32_t decoder_row;
} pes_parameters_t;

//! enum mapping region ids to regions in python
typedef enum pes_initalise {
    N_LEARNING_RULES, START_LEARNING_RULES
} pes_initalise;

//! counter for how many learning rules of pes there are
static uint32_t g_num_pes_learning_rules = 0;

//! the param tracker for pes learning rules
static pes_parameters_t *g_pes_learning_rules = NULL;

//! \brief When using non-filtered activity, applies PES to a spike vector
//! \param[in] ensemble
//! \param[in] modulatory_filters
void pes_apply(
        const ensemble_state_t *ensemble,
        const if_collection_t *modulatory_filters){
    // Extract parameters
    const ensemble_parameters_t *params = &ensemble->parameters;
    uint32_t n_neurons_total = params->n_neurons_total;
    uint32_t n_populations = params->n_populations;
    const uint32_t *pop_lengths = ensemble->population_lengths;
    value_t *decoder = ensemble->decoders;
    const uint32_t *spike_vector = ensemble->spikes;

    // Loop through all the learning rules
    for(uint32_t learning_rule = 0; learning_rule < g_num_pes_learning_rules;
            learning_rule++) {

        // If this learning rule operates on un-filtered activity and should,
        // therefore be updated here
        const pes_parameters_t *params = &g_pes_learning_rules[learning_rule];
        // Extract input signal from filter's output
        const if_filter_t *error_sig =
            &modulatory_filters->filters[params->error_sig_index];
        const value_t *error_val = error_sig->output;

        // Get pointer to first row of decoder matrix that this learning
        // rule modifies
        value_t *rule_decoder =
            &decoder[params->decoder_row * n_neurons_total];

        // For each population
        uint32_t decoder_col = 0;
        for (uint32_t p = 0; p < n_populations; p++){
            // Get the number of neurons in this population
            uint32_t pop_length = pop_lengths[p];

            // While we have neurons left to process
            while (pop_length){
                // Determine how many neurons are in the next word of the
                // spike vector.
                uint32_t n_neurons = (pop_length > 32) ? 32 : pop_length;

                // Load the next word of the spike vector
                uint32_t data = *(spike_vector++);

                // Include the contribution from each neuron
                while (n_neurons){  // While there are still neurons left
                    // Work out how many neurons we can skip
                    // XXX: The GCC documentation claims that
                    // `__builtin_clz(0)` is
                    // undefined, but the ARM instruction it uses is
                    // defined such that:
                    // CLZ 0x00000000 is 32
                    uint32_t skip = __builtin_clz(data);

                    // If `skip` is NOT less than `n` then there are
                    // either no firing
                    // neurons left in the word (`skip` == 32) or the
                    // first `1` in the word
                    // is beyond the range of bits we care about anyway.
                    if (skip < n_neurons){
                        // Skip until we reach the next neuron which fired
                        decoder_col += skip;

                        // Loop through output dimensions and apply PES
                        // learning
                        value_t *neuron_decoder =
                            &rule_decoder[decoder_col];
                        for(uint dimension = params->error_start_dim;
                            dimension < params->error_end_dim;
                            dimension++, neuron_decoder += n_neurons_total){
                            *neuron_decoder -=
                                (params->learning_rate *
                                 error_val[dimension]);
                        }

                        // Prepare to test the neuron after the one we
                        // just processed.
                        decoder_col++;
                        // Also skip the neuron we just decoded
                        skip++;
                        // Reduce the number of neurons left
                        pop_length -= skip;
                        // and the number left in this word.
                        n_neurons -= skip;
                        // Shift out processed neurons
                        data <<= skip;

                    }
                    // Otherwise, if there are no neurons left in
                    // this word
                    else{
                        // Point at the decoder for the next neuron
                        decoder_col += n_neurons;
                        // Reduce the number left in the population
                        pop_length -= n_neurons;
                        // No more neurons left to process
                        n_neurons = 0;
                    }
                }
            }
        }
    }
}

//! \brief initiaites a pes thing
//! \param[in] address: dsg address for pes
//! \return bool stating if the init succeeded or not.
bool pes_initialise(address_t address){
    // Read number of PES learning rules that are configured
    g_num_pes_learning_rules = address[N_LEARNING_RULES];

    log_info("PES learning: Num rules:%u\n", g_num_pes_learning_rules);

    if(g_num_pes_learning_rules > 0){
        // Allocate memory
        g_pes_learning_rules =
            spin1_malloc(g_num_pes_learning_rules * sizeof(pes_parameters_t));
        if( g_pes_learning_rules == NULL){
            log_error("failed to allocate dtcm for pes learning rules");
            return false;
        }

        // Copy learning rules from region into new array
        memcpy(g_pes_learning_rules, &address[START_LEARNING_RULES],
            g_num_pes_learning_rules * sizeof(pes_parameters_t));

        // Display debug
        for(uint32_t learning_rule = 0;
                learning_rule < g_num_pes_learning_rules; learning_rule++){
            const pes_parameters_t *parameters =
                &g_pes_learning_rules[learning_rule];
            log_debug(
                "\tRule %u, Learning rate:%k, Error signal index:%u, "
                "Error signal start dimension:%u, Error signal end "
                "dimensions:%u, Decoder row:%u\n",
                learning_rule, parameters->learning_rate,
                parameters->error_sig_index, parameters->error_start_dim,
                parameters->error_end_dim, parameters->decoder_row);
        }
    }
    return true;
}