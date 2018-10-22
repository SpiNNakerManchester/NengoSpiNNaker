#ifndef INPUT_FILTERING_H
#define INPUT_FILTERING_H

#include "spin1_api.h"
#include "common-typedefs.h"

//! An input accumulator.
typedef struct _if_accumulator_t
{
    value_t *value;  // Value of the accumulator
    uint32_t mask;   // Mask used to make the accumulator latching or otherwise
} if_accumulator_t;

//! \biref needed to allow function step store
typedef void (*FilterStep)(
    uint32_t n_dims, value_t *input, value_t *output, void *s);


//! A pair of input and output which are are joined by a filter function.
typedef struct _if_filter_t
{
    if_accumulator_t *input;  // Input accumulator
    value_t *output;          // Output value
    uint32_t size;            // Size of input and output vectors
    void *state;              // State maintained by the filter
    FilterStep step;          // Filter evaluation function
} if_filter_t;

//! \brief needed to allow function init store
typedef bool (*FilterInit)(
    address_t data_address, uint32_t offset, uint32_t *size_of_words_read,
    if_filter_t *filter, uint32_t size);

//! A pseudo routing table entry which can be used to determine which input a
//! packet should be included in.
typedef struct _if_route_t
{
    uint32_t key;   // Key against which to compare the received packet
    uint32_t mask;  // Mask against which to compare the received packet
    uint32_t dimension_mask;  // Mask to extract the index of the component
    uint32_t input_index; // Index of the input add the packet to
} if_route_t;

//! A collection of filters which share routing information (and possibly an
//! accumulated output value).
typedef struct _if_collection_t
{
    // Mandatory components
    uint32_t n_filters;  // Number of filters
    uint32_t n_routes;   // Number of routing entries
    if_filter_t *filters;  // Filters
    if_route_t *routes;    // Packet to filter routes

    // Optional components
    uint32_t output_size;  // Size of output vector (may be 0)
    value_t *output;       // Output vector (may be NULL)
} if_collection_t;

//! \brief Apply new or additional input to a filter.
//! \param[in] filter: the filter to process
//! \param[in] dimension: the width of the input
//! \param[in] value: ?????????
static inline void _if_filter_input(
    if_filter_t *filter, uint32_t dimension,  value_t value)
{
    // The new accumulator value for this filter is either the current value
    // plus the new value or just the new value depending on the value of the
    // mask.
    filter->input->value[dimension] = \
        kbits(bitsk(filter->input->value[dimension]) & filter->input->mask) + \
        value;
}

//! \brief Simulate one step of a filter and reset its accumulator if necessary
//! \param[in] filter: the filters to work with
static inline void _if_filter_step(if_filter_t* filter)
{
    // Disable interrupts to avoid a race condition
    uint32_t cpsr = spin1_fiq_disable();

    // Apply the simulation step
    filter->step(filter->size, filter->input->value,
                 filter->output, filter->state);

    // Apply the input accumulator step.  The mask will either set the
    // accumulator to zero or will leave it at its current value.
    for (uint32_t n = 0; n < filter->size; n++)
    {
        filter->input->value[n] =
            kbits(bitsk(filter->input->value[n]) & ~filter->input->mask);
    }

    // Re-enable interrupts
    spin1_mode_restore(cpsr);
}

//! \brief Returns true if the packet matched any routing entries, otherwise
//!        returns false.
//! \param[in] payload: ??????
//! \param[in] filters: the set of filters
//! \param[in] dim_offset: ????????
//! \param[in] key: ??????
//! \param[in] max_dim_sub_one: ????????
static inline bool input_filtering_input_with_dimension_offset(
        if_collection_t* filters, uint32_t key, uint32_t payload,
        uint32_t dim_offset, uint32_t max_dim_sub_one) {
    bool handled = false;

    // Look at all the routing entries, if we match an entry then include the
    // packet in the indicated input vector.
    for (uint32_t n = 0; n < filters->n_routes; n++)
    {
        // Get the routing entry and the filter referred to by the entry
        if_route_t route = filters->routes[n];
        if_filter_t *filter = &filters->filters[route.input_index];

        if ((key & route.mask) == route.key)
        {
            // Get the dimension of the packet
            // NOTE: if offset is 0 then the subtraction will be optimised out.
            const uint32_t dim = (key & route.dimension_mask) - dim_offset;

            // NOTE: If max_dim_sub_one is UINT32_MAX then the CMP is optimised
            // out as all packets will match.
            if (dim <= max_dim_sub_one)
            {
                // The packet matches this entry and is in the range of
                // dimensions expected; include the contribution from the
                // packet and indicate that we have handled the packet.
                _if_filter_input(filter, dim, kbits(payload));
                handled = true;
            }
        }
    }
    return handled;
}

//! \brief  Returns true if the packet matched any routing entries, otherwise
//!         returns false.
//! \param[in] filters: the set of filters
//! \param[in] key: ??????
//! \param[in] payload: ??????????
static inline bool input_filtering_input(
    if_collection_t* filters, uint32_t key, uint32_t payload) {
    // Input with no dimensional offset, the given arguments result in an
    // optimised version of the previous method being inlined.
    return input_filtering_input_with_dimension_offset(
        filters, key, payload, 0, UINT32_MAX);
}

//! \brief Apply all filter steps but DO NOT accumulate their outputs.
//! \param[in] filters: the filters to update
static inline void input_filtering_step_no_accumulate(
    if_collection_t *filters)
{
    // Apply the filter step for each filter in the collection.
    for (uint32_t n = filters->n_filters; n > 0; n--)
    {
        // Get the filter and apply the step function
        if_filter_t *filter = &filters->filters[n - 1];
        _if_filter_step(filter);
    }
}

//! \brief Apply all filter steps and accumulate their outputs.
//! \param[in] filters: the filters to update
static inline void input_filtering_step(
    if_collection_t *filters)
{
    // Zero the accumulator, not using memset as this would entail a further
    // function call.
    for (uint32_t d = filters->output_size; d > 0; d--)
    {
        filters->output[d - 1] = 0.0k;
    }

    // Apply all of the filter step functions and accumulate the outputs of the
    // filters.
    for (uint32_t n = filters->n_filters; n > 0; n--)
    {
        // Get the filter and apply the step function
        if_filter_t *filter = &filters->filters[n - 1];
        _if_filter_step(filter);

        // Get the filter output
        value_t *output = filters->filters[n - 1].output;

        // Include each dimension in turn
        for (uint32_t d = filters->output_size; d > 0; d--)
        {
            filters->output[d - 1] += output[d - 1];
        }
    }
}

//! \brief Copy in a set of routing entries.
//! \param[in] filters: the set of filters
//! \param[in] routes: array of if_routes.
//! \return bool stating if successfully initialised routes
bool input_filtering_initialise_routes(
    if_collection_t *filters, uint32_t *routes);

//! \brief creates input filters
//! \param[in] filter_output_array: where to store filters ouputs
//! \param[in] sdram_data: location in sdram where the filter data resides
//! \param[in] filters: where to store filters
//! \param[out] sdram_words_read: the number of words read during this init
//! \return bool stating if successfully initialised filters
bool input_filtering_initialise_filters(
    if_collection_t *filters, uint32_t *data, value_t **filter_output_array,
    uint32_t *sdram_words_read);

//! \brief Initialise a filter collection with an output accumulator.
//! \param[in] filters: set of filters
//! \param[in] n_dimensions: size of accumulator. Use zero to indicate that
//!                          no output accumulator should be assigned.
//! \return bool stating if successfully initialised output
bool input_filtering_initialise_output(
    if_collection_t *filters, uint32_t n_dimensions);


#endif //INPUT_FILTERING_H
