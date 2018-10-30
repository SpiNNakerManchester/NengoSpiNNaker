#include <common/nengo_typedefs.h>
#include <common/input_filtering.h>
#include <common/fixed_point.h>
#include <common/constants.h>
#include <string.h>
#include <debug.h>
#include "common-typedefs.h"

// Filter specification flags
#define LATCHING 1
#define LATCHING_MASK 0xFFFFFFFF
#define NOT_LATCHING_MASK 0x00000000

//! enum mapping region ids to regions in python
typedef enum filter_region_counter_positions {
    N_LOW_PASS_FILTERS, N_NONE_PASS_FILTERS, N_LINEAR_FILTERS,
    START_OF_LOW_PASS_FILTERS
} filter_region_counter_positions;

//! enum mapping the params n the basic filter params
typedef enum filter_region_basic_elements {
    // "Width" of the filter (number of dimensions)
    SIZE = 0,
    // Flags applied to the filter
    FLAGS = 1,
    // size read
    BASIC_FILTER_PARAMETER_SIZE = 2
} filter_region_basic_elements;

#define N_FILTER_TYPES 3

//! enum mapping params in region
typedef enum routing_region_paramter_positions{
    N_ROUTES, STARTS_OF_DATA
} routing_region_paramter_positions;

/* 1st Order Low-Pass ********************************************************/

/* General purpose LTI implementation ****************************************/
/* A Direct Form I implementation for linear digital filters.  This
 * implementation cause unnecessary numbers of pipeline flushes, so fixed order
 * filters can be implemented to reduce the cost of branch prediction failures.
 */

// Commonly used pair of value_t types
typedef struct _value_t_pair_t
{
  value_t a, b;
} value_t_pair_t;


typedef struct _lti_state_t
{
    uint32_t order;  // Order of the filter
    value_t_pair_t *abs;       // Filter coefficients

    // Previous values of the input and output. This is a 2D array, to access
    // the kth last value of dimension d of x use:
    //
    //     xyz[d*order + (n-k) % order].b
    //
    // For y use:
    //
    //     xyz[d*order + (n-k) % order].a
    value_t_pair_t *xyz;

    // We treat the previous values as two circular buffers. After a simulation
    // step `n` should be incremented (mod order) to rotate the ring.
    uint32_t n;
} lti_state_t;

//! linear filter internal parameters
struct _lti_filter_init_params {
    uint32_t order;
    value_t data;  // Array of parameters 2*order longer (a[...] || b[...])
} _lti_filter_init_params;



/* FILTER IMPLEMENTATIONS ****************************************************/
/* None filter : output = input **********************************************/
void _none_filter_step(uint32_t n_dims, value_t *input,
                       value_t *output, void *params){
    use(params);

    // The None filter just copies its input to the output
    for (uint32_t d = 0; d < n_dims; d++)
    {
        output[d] = input[d];
    }
}

//! \brief creates a None filter
//! \param[in] data_address: the sdram address where the filters data starts
//! \param[in] offset: position in sdram to move to to locate specific filter
//!                    data
//! \param[in] size_of_words_read: pointer for how many words read
//! \param[in] filter: the pointer to where to put filter data
//! \param[in] size: width of the filter
bool _none_filter_initialise(
        address_t data_address, uint32_t offset, uint32_t *size_of_words_read,
        if_filter_t *filter, uint32_t size){
    use(data_address);
    use(offset);
    use(size);
    use(size_of_words_read);

    log_debug(">> None filter\n");

    // We just need to set the function pointer for the step function.
    filter->step = _none_filter_step;
    size_of_words_read = 0;
    return true;
}


void _lowpass_filter_step(
      uint32_t n_dims, value_t *input, value_t *output, void *pars)
{
    // Cast the params
    value_t_pair_t *params = (value_t_pair_t *) pars;
    register int32_t a = bitsk(params->a);
    register int32_t b = bitsk(params->b);

    // Apply the filter to every dimension (realised as a Direct Form I digital
    // filter).
    for (uint32_t d = 0; d < n_dims; d++)
    {
        // The following is equivalent to:
        //
        //    output[d] *= params->a;
        //    output[d] += input[d] * params->b;

        // Compute the next value in a register
        register int64_t next_output;

        // Perform the first multiply
        int32_t current_output = bitsk(output[d]);
        next_output = __smull(current_output, a);

        // Perform the multiply accumulate
        int32_t current_input = bitsk(input[d]);
        next_output = __smlal(next_output, current_input, b);

        // Scale the result back down to store it
        output[d] = kbits(convert_s32_30_s16_15(next_output));
    }
}

//! \brief creates a low pass filter
//! \param[in] data_address: the sdram address where the filters data starts
//! \param[in] offset: position in sdram to move to to locate specific filter
//!                    data
//! \param[in] size_of_words_read: pointer for how many words read
//! \param[in] filter: the pointer to where to put filter data
//! \param[in] size: width of the filter
bool _lowpass_filter_initialise(
        address_t data_address, uint32_t offset, uint32_t *size_of_words_read,
        if_filter_t *filter, uint32_t size){
    use(size);

    // Copy the filter parameters into memory
    filter->state = spin1_malloc(sizeof(value_t_pair_t));
    if (filter->state == NULL){
        log_error("Cant allocate dtcm for filter state");
        return false;
    }
    spin1_memcpy(filter->state, (void*) data_address[offset],
                 sizeof(value_t_pair_t));

    log_debug(">> Lowpass filter (%k, %k)\n",
              ((value_t_pair_t *)filter->state)->a,
              ((value_t_pair_t *)filter->state)->b);

    // Store a reference to the step function
    filter->step = _lowpass_filter_step;
    *size_of_words_read =
        (int) (sizeof(value_t_pair_t) / BYTES_TO_WORD_CONVERSION);
    return true;
}

void _lti_filter_step(uint32_t n_dims, value_t *input,
                      value_t *output, void *s)
{
    // Cast the state
    lti_state_t *state = (lti_state_t *) s;

    // Apply the filter to every dimension (realised as a Direct Form I digital
    // filter).
    for (uint32_t d = n_dims, dd = n_dims - 1; d > 0; d--, dd--)
    {
        // Point to the correct previous x and y values.
        value_t_pair_t *xy = &state->xyz[dd * state->order];

        // Create the new output value for this dimension
        register int64_t output_val = 0;

        // Direct Form I filter
        // `m` acts as an index into the ring buffer of historic input and
        // output.
        for (uint32_t k=0, m = state->n; k < state->order; k++)
        {
            // Update the index into the ring buffer, if this would go
            // negative it wraps to the top of the buffer.
            if (m == 0)
            {
              m += state->order;
            }
            m--;

            // Apply this part of the filter
            // Equivalent to:
            //     output[dd] += ab.a * xyz.a;
            //     output[dd] += ab.b * xyz.b;
            value_t_pair_t ab = state->abs[k];
            value_t_pair_t xyz = xy[m];
            output_val = __smlal(output_val, bitsk(ab.a), bitsk(xyz.a));
            output_val = __smlal(output_val, bitsk(ab.b), bitsk(xyz.b));
        }

        // Include the initial new input
        xy[state->n].b = input[dd];

        // Save the current output for later steps
        output[dd] = kbits(convert_s32_30_s16_15(output_val));
        xy[state->n].a = output[dd];
    }

    // Rotate the ring buffer by moving the starting pointer, if the starting
    // pointer would go beyond the end of the buffer it is returned to the start.
    if (++state->n == state->order)
    {
        state->n = 0;
    }
}

//! \brief creates a linear filter
//! \param[in] data_address: the sdram address where the filters data starts
//! \param[in] offset: position in sdram to move to to locate specific filter
//!                    data
//! \param[in] size_of_words_read: pointer for how many words read
//! \param[in] filter: the pointer to where to put filter data
//! \param[in] size: width of the filter
bool _lti_filter_initialise(
        address_t data_address, uint32_t offset, uint32_t *size_of_words_read,
        if_filter_t *filter, uint32_t size){

    // Cast the parameters block
    struct _lti_filter_init_params *params = \
      (struct _lti_filter_init_params *) data_address[offset];
    *size_of_words_read =
        (int) (sizeof(_lti_filter_init_params) / BYTES_TO_WORD_CONVERSION);

    // Malloc space for the parameters
    filter->state = spin1_malloc(sizeof(lti_state_t));
    if(filter->state == NULL){
        log_error("Can not allocate DTCM for linear filter state");
        return false;
    }

    lti_state_t *state = (lti_state_t *) filter->state;
    state->order = params->order;

    log_debug(">> LTI Filter of order %d", state->order);
    state->abs = spin1_malloc(sizeof(value_t_pair_t) * state->order);
    if(state->abs == NULL){
        log_error("can not allocate DTCM for linear filter state->abs");
        return false;
    }

    // Malloc space for the state
    state->xyz = spin1_malloc(sizeof(value_t_pair_t) * state->order * size);
    if (state->xyz == NULL){
        log_error("Can not allocate dtcm for linear filter state->xyz");
        return false;
    }

    // Copy the parameters across
    value_t *data = &params->data;
    spin1_memcpy(state->abs, data, sizeof(value_t_pair_t) * state->order);
    size_of_words_read +=
        ((sizeof(value_t_pair_t) * state->order) * BYTES_TO_WORD_CONVERSION);

    // If debugging then print out all filter parameters
    for (uint32_t k = 0; k < state->order; k++)
    {
      log_debug("a[%d] = %k, b[%d] = %k\n", state->abs[k].a, state->abs[k].b);
    }

    // Zero all the state holding variables
    state->n = 0;
    memset(state->xyz, 0, sizeof(value_t_pair_t) * size * state->order);

    // Store a reference to the correct step function for the filter.
    // Insert any specially optimised filters here.
    filter->step = _lti_filter_step;
    return true;
}


//! \brief creates input filters
//! \param[in] filter_output_array: where to store filters outputs
//! \param[in] sdram_data: location in sdram where the filter data resides
//! \param[in] filters: where to store filters
//! \param[out] sdram_words_read: the number of words read during this init
//! \return bool stating if the initialisation was successful
bool input_filtering_initialise_filters(
        if_collection_t *filters, address_t sdram_data,
        value_t **filter_output_array, uint32_t *sdram_words_read) {
    // Get the number of filters and malloc sufficient space for the filter
    // parameters.
    use(*sdram_words_read);

    filters->n_filters = (
        sdram_data[N_LOW_PASS_FILTERS] + sdram_data[N_NONE_PASS_FILTERS] +
        sdram_data[N_LINEAR_FILTERS]);

    log_info("going to try to allocate dtcm for %d filters each with size %d "
             "bytes", filters->n_filters, sizeof(if_filter_t));
    log_info("got %d bytes left in heap", sark_heap_max(sark.heap, 0));

    filters->filters = spin1_malloc(filters->n_filters * sizeof(if_filter_t));
    if (filters->filters == NULL){
        log_error("Cannot allocate the filters DTCM of %d bytes for %d "
        "filters", filters->n_filters * sizeof(if_filter_t),
                   filters->n_filters);
        return false;
    }

    log_info(
        "Loading %d low pass filters, %d none pass filters and %d linear "
        "filters\n",
        sdram_data[N_LOW_PASS_FILTERS], sdram_data[N_NONE_PASS_FILTERS],
        sdram_data[N_LINEAR_FILTERS]);

    log_info("god damn you chimp");

    // Map of filter indices to filter initialisation methods
    FilterInit filter_types_init[] = {
        _lowpass_filter_initialise,
        _none_filter_initialise,
        _lti_filter_initialise,
    };

    log_info("damn you mundy");

    // map of filter types to n filters
    uint32_t n_filters[] = {
        sdram_data[N_LOW_PASS_FILTERS],
        sdram_data[N_NONE_PASS_FILTERS],
        sdram_data[N_LINEAR_FILTERS],
    };

    log_info("damn you mundy again");

    log_info("damn you mundy yet again");

    // process low pass filters
    uint32_t data_index = START_OF_LOW_PASS_FILTERS;
    for (uint32_t filter_type_index=0; filter_type_index<N_FILTER_TYPES;
         filter_type_index++){

        log_info("processing filter in type index %d", filter_type_index);

        for (uint32_t filter_index = 0;
             filter_index < n_filters[filter_type_index];
             filter_index++) {

            log_info("im a pirate for filter index %d", filter_index);

            // Get the size of the filter, store it
            filters->filters[filter_index].size = sdram_data[data_index + SIZE];

            log_info(
                "> Filter [%d] size = %d\n",
                filter_index, sdram_data[data_index + SIZE]);

            // Initialise the input accumulator
            log_info("trying to allocate filter dtcm");
            filters->filters[filter_index].input = spin1_malloc(
                sizeof(if_accumulator_t));
            if (filters->filters[filter_index].input == NULL) {
                log_error("Failed to allocate filter DTCM memory");
                return false;
            }

            // Initialise the input accumulator value
            log_info("trying input value");
            filters->filters[filter_index].input->value = spin1_malloc(
                sizeof(value_t) * sdram_data[data_index + SIZE]);
            if (filters->filters[filter_index].input->value == NULL) {
                log_error(
                    "Failed to allocate dtcm for filters input values. "
                    "wanted to allocate %d bytes for size %d",
                    sizeof(value_t) * sdram_data[data_index + SIZE],
                    sdram_data[data_index + SIZE]);
                return false;
            }

            log_info("you smell");

            // process latching
            if (sdram_data[data_index + FLAGS] == LATCHING) {
                filters->filters[filter_index].input->mask = LATCHING_MASK;
            } else {
                filters->filters[filter_index].input->mask = NOT_LATCHING_MASK;
            }

            log_info("you smell worse");

            // process output
            if (filter_output_array == NULL) {
                log_info("you smell as bad as sergio");
                filters->filters[filter_index].output = spin1_malloc(
                    sizeof(value_t) * sdram_data[data_index + SIZE]);
                if (filters->filters[filter_index].output == NULL) {
                    log_error(
                        "Failed to allocate filter %d output", filter_index);
                    return false;
                }
            } else { // Otherwise, copy output pointer into filter
                log_info("you smell asa chimpy");
                filters->filters[filter_index].output =
                    filter_output_array[filter_index];

            }

            log_info("you smell as andy");

            // Zero the input and the output
            memset(filters->filters[filter_index].input->value, 0,
                   sizeof(value_t) * sdram_data[data_index + SIZE]);
            log_info("you just smell overall");
            memset(filters->filters[filter_index].output, 0,
                   sizeof(value_t) * sdram_data[data_index + SIZE]);
            log_info("god damn you smell");

            // pointer for tracking number of words a filter reads
            uint32_t size_of_words_read = 0;

            if(!filter_types_init[filter_type_index](
                sdram_data, data_index + BASIC_FILTER_PARAMETER_SIZE,
                &size_of_words_read, &filters->filters[filter_index],
                sdram_data[data_index + SIZE])){
                log_error("failed to instantiate filter");
                return false;
            }
            data_index += BASIC_FILTER_PARAMETER_SIZE + size_of_words_read;
        }
    }
    *sdram_words_read = data_index;
    return true;
}

//! \brief Copy in a set of routing entries.
//! \param[in] filters: the set of filters
//! \param[in] address: array of if_routes.
//! \param[out] sdram_words_read: the number of words read during this init
//! \return bool that states if the initialisation succeeds.
bool input_filtering_initialise_routes(
        if_collection_t *filters, uint32_t *address,
        uint32_t *sdram_words_read){

    use(sdram_words_read);

    // Copy in the number of routing entries
    filters->n_routes = address[N_ROUTES];
    log_info("Loading %d filter routes\n", filters->n_routes);

    // Malloc sufficient room for the entries
    filters->routes = spin1_malloc(filters->n_routes * sizeof(if_route_t));
    if (filters->routes == NULL){
        log_error("failed to malloc filter routes");
        return false;
    }

    // Copy the entries across
    spin1_memcpy(filters->routes, &address[STARTS_OF_DATA],
                 filters->n_routes * sizeof(if_route_t));

    // update n words written
    *sdram_words_read = STARTS_OF_DATA + (
        (filters->n_routes * sizeof(if_route_t)) / BYTES_TO_WORD_CONVERSION);

    // debug print
    for (uint32_t n = 0; n < filters->n_routes; n++)
    {
        log_debug("\tRoute[%d] = (0x%08x, 0x%08x) dmask=0x%08x => %d\n",
              n, filters->routes[n].key, filters->routes[n].mask,
              filters->routes[n].dimension_mask,
              filters->routes[n].input_index);
    }

    return true;
}

//! \brief Initialise a filter collection with an output accumulator.
//! \param[in] filters: set of filters
//! \param[in] n_dimensions: size of accumulator. Use zero to indicate that
//!                          no output accumulator should be assigned.
bool input_filtering_initialise_output(if_collection_t *filters,
                                       uint32_t n_dimensions){
  // Store the output size
  filters->output_size = n_dimensions;

  // If the output size is zero then don't allocate an accumulator, otherwise
  // malloc sufficient space.
  //TODO me thinks this if is pointless. only else can exist
  if (n_dimensions == 0)
  {
    filters->output = NULL;
  }
  else
  {
    filters->output = spin1_malloc(sizeof(value_t) * n_dimensions);
    if (filters->output == NULL){
        return false;
    }
  }
  return true;
}


