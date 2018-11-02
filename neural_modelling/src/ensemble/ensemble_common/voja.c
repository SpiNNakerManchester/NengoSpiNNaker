#include "voja.h"
#include <stdfix-full-iso.h>
#include <debug.h>

//! enum mapping region ids to regions in python
typedef enum voja_initialize {
    N_LEARNING_RULES, ONE_OVER_RADIUS, START_OF_VOJA_RULES
} voja_initialize;

//! number of voja learning rules
uint32_t g_num_voja_learning_rules = 0;

//! param 1/radius
value_t g_voja_one_over_radius = CONSTANT_ERROR_RATE;

//! set of voja learning rules parameters
voja_parameters_t *g_voja_learning_rules = NULL;


//! \brief Copy in data controlling the Voja learning
//!        rule from the Voja region of the Ensemble.
//! \param[in] address: the dsg address for the voja region
bool voja_initialise(address_t address){
    // Read number of Voja learning rules that are configured and the scaling
    // factor
    g_num_voja_learning_rules = address[N_LEARNING_RULES];
    g_voja_one_over_radius = kbits(address[ONE_OVER_RADIUS]);

    log_info(
        "Voja learning: Num rules:%u, One over radius:%k\n",
        g_num_voja_learning_rules, g_voja_one_over_radius);

    if(g_num_voja_learning_rules > 0){
        // Allocate memory
        g_voja_learning_rules = spin1_malloc(
            g_num_voja_learning_rules * sizeof(voja_parameters_t));
        if (g_voja_learning_rules == NULL){
            log_error("failed to allocate dtcm for voja rules");
            return false;
        }

        // Copy learning rules from region into new array
        spin1_memcpy(g_voja_learning_rules,
            &address[START_OF_VOJA_RULES],
            g_num_voja_learning_rules * sizeof(voja_parameters_t));

        // Display debug
        for(uint32_t learning_rule = 0;
                learning_rule < g_num_voja_learning_rules; learning_rule++){
            const voja_parameters_t *parameters =
                &g_voja_learning_rules[learning_rule];
            log_info(
                "\tRule %u, Learning rate:%k, Learning signal filter "
                "index:%d, Encoder output offset:%u, Decoded input filter "
                "index:%u\n",
                learning_rule, parameters->learning_rate,
                parameters->learning_signal_filter_index,
                parameters->encoder_offset,
                parameters->decoded_input_filter_index);
        }
    }
    return true;
}