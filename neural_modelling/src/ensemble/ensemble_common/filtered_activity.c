#include "filtered_activity.h"
#include <debug.h>

//! initial state for the filtered activities filter
#define INITIAL_STATE 0

//! enum covering data region elements
typedef enum filtered_activity_region_elements {
    N_FILTERS, START_OF_FILTERS
} filtered_activity_region_elements;

//! number of filters
uint32_t g_num_activity_filters = 0;

//! filter activities??????
value_t **g_filtered_activities = NULL;

//! filter parameters
activity_filter_parameters_t *g_activity_filter_params = NULL;

//! \brief inits the filtered activity
//! \param[in] address: sdram address of init data
//! \param[in] n_neurons: the number of neurons to init for
//! \return bool that states if init was successful or not
bool filtered_activity_initialise(address_t address, uint32_t n_neurons)
{
    // Read number of PES learning rules that are configured
    g_num_activity_filters = address[N_FILTERS];

    log_info("Filtered activity: Num filters:%u\n", g_num_activity_filters);

    if(g_num_activity_filters > 0)
    {
        // allocate dtcm as required
        g_activity_filter_params = spin1_malloc(
            g_num_activity_filters * sizeof(activity_filter_parameters_t));
        if (g_activity_filter_params == NULL){
            log_error("failed to allocate dtcm for filtered activity filters");
            return false;
        }
        g_filtered_activities = spin1_malloc(
            g_num_activity_filters * sizeof(value_t*));
        if (g_filtered_activities == NULL){
            log_error("failed to allocate dtcm for filtered activities");
            return false;
        }

        // Copy propogators from region into new array
        memcpy(g_activity_filter_params, &address[START_OF_FILTERS],
               g_num_activity_filters * sizeof(activity_filter_parameters_t));

        // Loop through filters
        for(uint32_t current_filter = 0;
                current_filter < g_num_activity_filters; current_filter++)
        {
            log_debug(
                "\tFilter %u, Filter:%k, 1.0 - Filter:%f\n",
                current_filter, g_activity_filter_params[current_filter].filter,
                g_activity_filter_params[current_filter].n_filter);

            // Allocate per-neuron filtered g_filtered_activities
            g_filtered_activities[current_filter] =
                spin1_malloc( n_neurons * sizeof(value_t));
            if (g_filtered_activities[current_filter] == NULL){
                log_error("failed to allocate dtcm for current filter");
                return false;
            }

            // Initially zero all filters
            memset(g_filtered_activities[current_filter], INITIAL_STATE,
                   n_neurons * sizeof(value_t));
        }
    }
    return true;
}

//! \brief process the filters for a time step
//! \param[in] n_neurons: the number of neurons to process.
void filtered_activity_step(uint32_t n_neurons)
{
    // Loop through filters
    for(uint32_t f = 0; f < g_num_activity_filters; f++)
    {
        // Loop through neurons and apply propogators
        for(uint32_t n = 0; n < n_neurons; n++)
        {
            g_filtered_activities[f][n] *= g_activity_filter_params[f].filter;
        }
    }
}