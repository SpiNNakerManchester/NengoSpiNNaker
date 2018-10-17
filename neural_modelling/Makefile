BUILDS = value_source value_sink sdp_transmitter sdp_receiver 
DIRS = $(BUILDS:%=makefiles/%)

all: $(DIRS)
	for d in $(DIRS); do $(MAKE) -C $$d || exit $$?; done

clean: $(DIRS)
	for d in $(DIRS); do $(MAKE) -C $$d clean || exit $$?; done

