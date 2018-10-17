#ifndef __SLOTS_H__
#define __SLOTS_H__

#include <debug.h>
#include <spin1_api.h>

bool _dma_complete = false;
uint slots_dma_port = 0;

typedef struct __slot_t {
    uint* data;
    uint current_pos;
    uint length;
} _slot_t;

typedef struct _slots_t {
    _slot_t* current; //!< The current slot
    _slot_t* next;    //!< The next slot
    _slot_t slots[2];
} slots_t;

static inline void slots_dma_complete(uint unused, uint unused2){
    use(unused);
    use(unused2);
    _dma_complete = true;
}

static inline bool initialise_slots(slots_t* slots, uint size, uint dma_port) {
    // Initialise the slots with the given size
    for (uint i = 0; i < 2; i++) {
        slots->slots[i].data = (uint*) spin1_malloc(size);
        if (slots->slots[i].data == NULL){
            log_error("Failed to malloc " #VAR " (%d bytes of DTCM)\n", size);
        }
        slots->slots[i].current_pos = 0;
        slots->slots[i].length = 0;
    }

    slots->current = &slots->slots[0];
    slots->next = &slots->slots[1];

    simulation_dma_transfer_done_callback_on(dma_port, slots_dma_complete);
    slots_dma_port = dma_port;
    return true;
}

static inline void slots_progress(slots_t* slots) {
  while(!_dma_complete){
      // do nothing, will be interrupted by the dma complete
  }

  // Swap the slots pointers
  _slot_t* t = slots->next;
  slots->next = slots->current;
  slots->current = t;

  // Clear the new next slot
  slots->next->length = 0;
  slots->next->current_pos = 0;
}

static inline void slots_set_up_dma(
        void *system_address, void *tcm_address,
        uint direction, uint length){
    spin1_dma_transfer(slots_dma_port, system_address, tcm_address, direction,
                       length);
    _dma_complete = false;
}


#endif
