#ifndef NEURAL_MODELLING_ENSEMBLE_H
#define NEURAL_MODELLING_ENSEMBLE_H

//! Parameters for the locally represented neurons this is all data stored
//! within the system region.
typedef struct ensemble_parameters{
    // Number of neurons in this portion
    uint32_t n_neurons;

    // Number of dimensions represented
    uint32_t n_dims;

    // Total width of encoder
    uint32_t encoder_width;

    // Number of neurons overall
    uint32_t n_neurons_total;

    // Number of neurons overall
    uint32_t n_populations;

    // Index of this population
    uint32_t population_id;

    // Index of first dimension
    uint32_t input_subspace_offset;

    // Number of dimensions
    uint32_t input_subspace_n_dims;

    // Number of output dimensions
    uint32_t n_decoder_rows;

    // Number of learnt output dimensions
    uint32_t n_learnt_decoder_rows;

    // Pointer into SDRAM for the input vector
    value_t *sdram_input_vector;

    // Our portion of the shared input vector
    value_t *sdram_input_vector_local;

    // Pointer into SDRAM for the spike vector
    uint32_t *sdram_spike_vector;

    // Our portion of the shared spike vector
    uint32_t *sdram_spikes_vector_local;

    // semaphore base sdram address
    uint32_t semaphore_base_address;

    // Number of learnt input signals
    uint32_t n_learnt_input_signals;

} ensemble_parameters_t;

//! state for the ensemble
typedef struct ensemble_state{
    // Generic parameters
    ensemble_parameters_t parameters;

    // Neuron state
    void *state;

    // Filtered input vector
    value_t *input;

    // Start of the section of input we update
    value_t *input_local;

    // Filtered input vector from each signal
    value_t **learnt_input;

    // Start of section of learnt_input we update
    value_t **learnt_input_local;

    // Globally inhibitory input
    value_t inhibitory_input;

    // Encoder matrix
    value_t *encoders;

    // Neuron biases
    value_t *bias;

    // Neuron gains
    value_t *gain;

    // Lengths of all populations
    uint32_t *population_lengths;

    // Length of padded spike vector (words)
    uint32_t sdram_spikes_length;

    // Unpadded spike vector
    uint32_t *spikes;

    // Rows from the decoder matrix
    value_t *decoders;

    // Output keys
    uint32_t *keys;
} ensemble_state_t;

#endif //NEURAL_MODELLING_ENSEMBLE_H
