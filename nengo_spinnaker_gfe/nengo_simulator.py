import logging
import os
import sys

import numpy

from nengo.cache import NoDecoderCache
from nengo.probe import Probe as NengoProbe

from nengo_spinnaker_gfe.abstracts.abstract_probeable import AbstractProbeable
from nengo_spinnaker_gfe.utility_objects.nengo_machine_graph_generator import \
    NengoMachineGraphGenerator
from nengo_spinnaker_gfe import binaries, constants
from nengo_spinnaker_gfe.utility_objects.\
    nengo_application_graph_generator import NengoApplicationGraphGenerator
from nengo_spinnaker_gfe import overridden_mapping_algorithms
from pacman.executor.injection_decorator import provide_injectables, \
    clear_injectables

from spinnaker_graph_front_end.spinnaker import SpiNNaker

from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.utilities.utility_objs import ExecutableFinder


logger = logging.getLogger(__name__)


class NengoSimulator(SpiNNaker):
    """SpiNNaker simulator for Nengo models.

    The simulator period determines how much data will be stored on SpiNNaker
    and is the maximum length of simulation allowed before data is transferred
    between the machine and the host PC. 
    
    For any other value simulation lengths of less than or equal to the period
    will be in real-time, longer simulations will be possible but will include
    short gaps when data is transferred between SpiNNaker and the host.

    :py:meth:`~.close` should be called when the simulator will no longer be
    used. This will close all sockets used to communicate with the SpiNNaker
    machine and will leave the machine in a clean state. Failure to call
    `close` may result in later failures. Alternatively `with` may be used::

        sim = nengo_spinnaker_gfe.Simulator(network)
        with sim:
            sim.run(10.0)
    """

    __slots__ = [
        "_nengo_operator_graph",
        "_nengo_object_to_data_map",
        "_profiled_nengo_object_to_data_map",
        "_nengo_to_app_graph_map",
        "_nengo_app_machine_graph_mapper",
        "_app_graph_to_nengo_operator_map"
    ]

    CONFIG_FILE_NAME = "nengo_spinnaker.cfg"
    NENGO_ALGORITHM_XML_FILE_NAME = "nengo_overridden_mapping_algorithms.xml"
    NENGO_PARTITIONER_ALGORITHM = "NengoPartitioner"

    def __init__(
            self, network, dt=constants.DEFAULT_DT,
            time_scale=constants.DEFAULT_TIME_SCALE,
            host_name=None, graph_label=None,
            database_socket_addresses=None, dsg_algorithm=None,
            n_chips_required=None, extra_pre_run_algorithms=None,
            extra_post_run_algorithms=None, decoder_cache=NoDecoderCache(),
            function_of_time_nodes=None,
            function_of_time_nodes_time_period=None):
        """Create a new Simulator with the given network.
        
        :param time_scale: Scaling factor to apply to the simulation, e.g.,\
            a value of `0.5` will cause the simulation to run at twice \
            real-time.
        :type time_scale: float
        :param host_name: Hostname of the SpiNNaker machine to use; if None\  
            then the machine specified in the config file will be used.
        :type host_name: basestring or None
        :param dt: The length of a simulator timestep, in seconds.
        :type dt: float
        :param graph_label: human readable graph label
        :type graph_label: basestring
        :param database_socket_addresses:
        :type database_socket_addresses:
        :param dsg_algorithm:
        :type dsg_algorithm:
        :param n_chips_required:
        :type n_chips_required:
        :param extra_post_run_algorithms:
        :type extra_post_run_algorithms:
        :param extra_pre_run_algorithms:
        :type extra_pre_run_algorithms:
        values
        :rtype None
        """
        self._nengo_object_to_data_map = dict()
        self._profiled_nengo_object_to_data_map = dict()
        self._nengo_to_app_graph_map = None
        self._app_graph_to_nengo_operator_map = None
        self._nengo_app_machine_graph_mapper = None

        executable_finder = ExecutableFinder()
        executable_finder.add_path(os.path.dirname(binaries.__file__))

        # Calculate the machine timestep, this is measured in microseconds
        # (hence the 1e6 scaling factor).
        machine_time_step = (
            int((dt / time_scale) *
                constants.SECONDS_TO_MICRO_SECONDS_CONVERTER))

        xml_paths = list()
        xml_paths.append(os.path.join(os.path.dirname(
            overridden_mapping_algorithms.__file__),
            self.NENGO_ALGORITHM_XML_FILE_NAME))

        SpiNNaker.__init__(
            self, executable_finder, host_name=host_name,
            graph_label=graph_label,
            database_socket_addresses=database_socket_addresses,
            dsg_algorithm=dsg_algorithm,
            n_chips_required=n_chips_required,
            extra_pre_run_algorithms=extra_pre_run_algorithms,
            extra_post_run_algorithms=extra_post_run_algorithms,
            time_scale_factor=time_scale,
            default_config_paths=[(
                os.path.join(os.path.dirname(__file__),
                             self.CONFIG_FILE_NAME))],
            machine_time_step=machine_time_step,
            extra_xml_paths=xml_paths)

        # basic mapping extras
        extra_mapping_algorithms = [
            "NengoKeyAllocator", "NengoHostGraphUpdateBuilder",
            "NengoCreateHostSimulator", "NengoSetUpLiveIO"]

        # only add the sdram edge allocator if not using a virtual board
        if not helpful_functions.read_config_boolean(
                self.config, "Machine", "virtual_board"):
            extra_mapping_algorithms.append(
                "NengoSDRAMOutgoingPartitionAllocator")

        if function_of_time_nodes is None:
            function_of_time_nodes = list()
        if function_of_time_nodes_time_period is None:
            function_of_time_nodes_time_period = list()

        # update the main flow with new algorithms and params
        self.extend_extra_mapping_algorithms(extra_mapping_algorithms)
        self.update_extra_inputs(
            {'NengoNodesAsFunctionOfTime': function_of_time_nodes,
             'NengoNodesAsFunctionOfTimeTimePeriod':
                 function_of_time_nodes_time_period,
             'NengoModel': network,
             'NengoDecoderCache': decoder_cache,
             "NengoNodeIOSetting": self.config.get("Simulator", "node_io"),
             "NengoEnsembleProfile":
                 self.config.getboolean("Ensemble", "profile"),
             "NengoEnsembleProfileNumSamples":
                 helpful_functions.read_config_int(
                     self.config, "Ensemble", "profile_num_samples"),
             "NengoRandomNumberGeneratorSeed":
                helpful_functions.read_config_int(
                    self.config, "Simulator", "global_seed"),
             "NengoUtiliseExtraCoreForProbes":
                self.config.getboolean(
                    "Node", "utilise_extra_core_for_probes"),
             "MachineTimeStepInSeconds": dt,
             "ReceiveBufferPort": helpful_functions.read_config_int(
                self.config, "Buffers", "receive_buffer_port"),
             "ReceiveBufferHost": self.config.get(
                 "Buffers", "receive_buffer_host"),
             "MinBufferSize": self.config.getint(
                 "Buffers", "minimum_buffer_sdram"),
             "MaxSinkBuffingSize": self.config.getint(
                 "Buffers", "sink_vertex_max_sdram_for_buffing"),
             "UsingAutoPauseAndResume": self.config.getboolean(
                 "Buffers", "use_auto_pause_and_resume"),
             "TimeBetweenRequests": self.config.getint(
                 "Buffers", "time_between_requests"),
             "BufferSizeBeforeReceive": self.config.getint(
                 "Buffers", "buffer_size_before_receive"),
             "SpikeBufferMaxSize": self.config.getint(
                "Buffers", "spike_buffer_size"),
             "VariableBufferMaxSize": self.config.getint(
                "Buffers", "variable_buffer_size")})

        # build app graph, machine graph, as the main tools expect an
        # application / machine graph level, and cannot go from random to app
        #  graph.
        nengo_app_graph_generator = NengoApplicationGraphGenerator()

        (self._nengo_operator_graph, host_network,
         self._nengo_to_app_graph_map, self._app_graph_to_nengo_operator_map,
         random_number_generator) = \
            nengo_app_graph_generator(
            self._extra_inputs["NengoModel"], self.machine_time_step,
            self._extra_inputs["NengoRandomNumberGeneratorSeed"],
            self._extra_inputs["NengoDecoderCache"],
            self._extra_inputs["NengoUtiliseExtraCoreForProbes"],
            self._extra_inputs["NengoNodesAsFunctionOfTime"],
            self._extra_inputs["NengoNodesAsFunctionOfTimeTimePeriod"],
            self.config.getboolean("Node", "optimise_utilise_interposers"),
            self._print_timings, self._do_timings, self._xml_paths,
            self._pacman_executor_provenance_path,
            self._extra_inputs["NengoEnsembleProfile"],
            self._extra_inputs["NengoEnsembleProfileNumSamples"],
            self._extra_inputs["ReceiveBufferPort"],
            self._extra_inputs["ReceiveBufferHost"],
            self._extra_inputs["MinBufferSize"],
            self._extra_inputs["MaxSinkBuffingSize"],
            self._extra_inputs["UsingAutoPauseAndResume"],
            self._extra_inputs["TimeBetweenRequests"],
            self._extra_inputs["BufferSizeBeforeReceive"],
            self._extra_inputs["SpikeBufferMaxSize"],
            self._extra_inputs["VariableBufferMaxSize"],
            self._extra_inputs["MachineTimeStepInSeconds"])

        # add the extra outputs as new inputs
        self.update_extra_inputs(
            {"NengoHostGraph": host_network,
             "NengoGraphToAppGraphMap": self._nengo_to_app_graph_map,
             "AppGraphToNengoOperatorMap":
                 self._app_graph_to_nengo_operator_map,
             "NengoRandomNumberGenerator": random_number_generator,
             "NengoOperatorGraph": self._nengo_operator_graph})

    def __enter__(self):
        """Enter a context which will close the simulator when exited."""
        # Return self to allow usage like:
        #
        #     with nengo_spinnaker_gfe.Simulator(model) as sim:
        #         sim.run(1.0)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Exit a context and close the simulator."""
        self.close()

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        # Determine how many steps to simulate for
        steps = int(
            numpy.round(
                float((time_in_seconds *
                       constants.SECONDS_TO_MICRO_SECONDS_CONVERTER))
                / self.machine_time_step))
        self._run_steps(steps)

    def _run_steps(self, steps):
        """Simulate for the given number of steps."""

        self._generate_machine_graph(steps)
        SpiNNaker.run(self, steps)

        # extract data
        try:
            self._extract_data()
        except Exception as e:
            e_inf = sys.exc_info()
            self._recover_from_error(
                e, e_inf, self.get_generated_output("ExecutableTargets"))

    @property
    def data(self):
        return self._nengo_object_to_data_map

    def _extract_data(self):
        """ extracts data from machine (only probes and ensembles so far)
        
        :return: 
        """

        # provide the outputs to the injector scope
        provide_injectables(self._last_run_outputs)

        # go through app verts looking for specific vertex types
        for application_vertex in self._nengo_operator_graph.vertices:

            # probe based data items
            if isinstance(application_vertex, AbstractProbeable):

                # tie in to data map
                nengo_probes = \
                    self._app_graph_to_nengo_operator_map[application_vertex]
                self._process_data_from_probe(nengo_probes, application_vertex)

        # clean injectable scope
        clear_injectables()

    def _process_data_from_probe(self, nengo_probes, application_vertex):
        for nengo_probe in nengo_probes:
            if isinstance(nengo_probe, NengoProbe):
                data = application_vertex.get_data_for_variable(
                    variable=nengo_probe.attr,
                    run_time=self.get_generated_output("RunTime"),
                    placements=(self.get_generated_output("MemoryPlacements")),
                    graph_mapper=self._nengo_app_machine_graph_mapper,
                    buffer_manager=(self.get_generated_output("BufferManager")))

                # add data to the sim store for probe data
                if nengo_probe in self._nengo_object_to_data_map:
                    self._nengo_object_to_data_map[nengo_probe] = \
                        numpy.vstack((
                            self._nengo_object_to_data_map[nengo_probe], data))
                else:
                    self._nengo_object_to_data_map[nengo_probe] = data

    def _generate_machine_graph(self, steps):
        """ generate the machine graph in context of pre allocated system 
        resoruces
        
        :param steps: 
        :return: 
        """

        self._get_max_available_machine()

        # get pre alloc res for allowing graph partitioning to operate correctly
        (system_pre_alloc_res_inputs,
         system_pre_alloc_res_algorithms) = \
            self._get_system_functionality_algorithms_and_inputs()

        machine_graph_generator = NengoMachineGraphGenerator()
        executor_items = machine_graph_generator(
            system_pre_allocated_resources_inputs=(
                system_pre_alloc_res_inputs),
            max_machine_outputs=self._max_machine_outputs,
            max_machine_available=self._max_machine_available,
            steps=steps,
            partitioning_algorithm=self.NENGO_PARTITIONER_ALGORITHM,
            system_pre_alloc_res_algorithms=system_pre_alloc_res_algorithms,
            print_timings=self._print_timings,
            do_timings=self._do_timings,
            nengo_operator_graph=self._nengo_operator_graph,
            xml_paths=self._xml_paths,
            machine_time_step=self._machine_time_step,
            pacman_executor_provenance_path=(
                self._pacman_executor_provenance_path),
            first_machine_time_step=self._current_run_timesteps,
            machine_time_step_in_seconds=(
                self._extra_inputs["MachineTimeStepInSeconds"]),
            )

        self._original_machine_graph = executor_items.get("MemoryMachineGraph")
        self._nengo_app_machine_graph_mapper = \
            executor_items.get("NengoGraphMapper")
        self._max_machine_outputs = executor_items

        # if the machine is set, then not gone though spalloc, so the new max
        # machine outputs are also the new machine outputs.
        if self._machine is not None:
            self._machine_outputs = self._max_machine_outputs

        self.update_extra_mapping_inputs(
            {"NengoGraphMapper": self._nengo_app_machine_graph_mapper})

    def close(self, turn_off_machine=None, clear_routing_tables=None,
              clear_tags=None):
        """Clean the SpiNNaker board and prevent further simulation."""
        if not self._closed:
            self.io_controller.close()
            self.controller.send_signal("stop")
            SpiNNaker.stop(
                self=self, turn_off_machine=turn_off_machine,
                clear_tags=clear_tags,
                clear_routing_tables=clear_routing_tables)
