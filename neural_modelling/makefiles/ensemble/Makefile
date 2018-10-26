APP = ensemble
SOURCES = ensemble/ensemble.c \
          ensemble/ensemble_common/pes.c  \
          ensemble/ensemble_common/voja.c \
          ensemble/ensemble_common/neuron_lif.c \
          common/input_filtering.c \
          common/out_spikes.c

CFLAGS += -I $(MODIFIED_DIR)/common/ $(MODIFIED_DIR)/ensemble/ensemble_common
include ../nengo_support.mk