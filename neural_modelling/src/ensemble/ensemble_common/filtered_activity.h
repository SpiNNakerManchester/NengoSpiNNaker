#ifndef NEURAL_MODELLING_FILTERED_ACTIVITY_H
#define NEURAL_MODELLING_FILTERED_ACTIVITY_H

//! params for a activity filter
typedef struct activity_filter_parameters_t
{
    // Filter value, e.g., \f$\exp(-\frac{dt}{\tau})\f$
    value_t filter;

    // 1 - filter value
    value_t n_filter;
} activity_filter_parameters_t;

//! number of filters
extern uint32_t g_num_activity_filters;

//! filter activities??????
extern value_t **g_filtered_activities;

//! filter parameters
activity_filter_parameters_t *g_activity_filter_params;


//! \brief apply effect of neuron spiking to all filtered activities
//! \param[in] neuron_id: neuron id that fired
static inline void filtered_activity_neuron_spiked(uint32_t neuron_id)
{
    // Loop through filters and add n_filter to activites
    for(uint32_t current_filter = 0; current_filter < g_num_activity_filters;
            current_filter++){
        g_filtered_activities[current_filter][neuron_id] +=
            g_activity_filter_params[current_filter].n_filter;
    }
}

//! \brief Copy in data controlling filtered activities from the filtered
//!        activity region of the Ensemble.
//! \param[in] address: sdram address of init data
//! \param[in] n_neurons: the number of neurons to init for
//! \return bool that states if init was successful or not
bool filtered_activity_initialise(address_t address, uint32_t n_neurons);

//! \brief Apply decay to all filtered activities
void filtered_activity_step();

#endif //NEURAL_MODELLING_FILTERED_ACTIVITY_H
