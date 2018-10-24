#ifndef NEURAL_MODELLING_PES_H
#define NEURAL_MODELLING_PES_H

// Common includes
#include "input_filtering.h"

//! \brief When using non-filtered activity, applies PES to a spike vector
//! \param[in] ensemble
//! \param[in] modulatory_filters
void pes_apply(
    const ensemble_state_t *ensemble,
    const if_collection_t *modulatory_filters);

//! \brief Copy in data controlling the PES learning rule from the PES
//!        region of the Ensemble.
//! \param[in] address: dsg address for pes
//! \return bool stating if the init succeeded or not.
bool pes_initialise(address_t address);



#endif //NEURAL_MODELLING_PES_H
