BUILDS = value_source value_sink sdp_transmitter
DIRS = $(BUILDS:%=makefiles/%)

all: $(DIRS)
	for d in $(DIRS); do $(MAKE) -C $$d || exit $$?; done

clean: $(DIRS)
	for d in $(DIRS); do $(MAKE) -C $$d clean || exit $$?; done

