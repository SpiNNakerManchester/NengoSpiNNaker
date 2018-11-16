from pacman.executor import PACMANAlgorithmExecutor


class NengoApplicationGraphGenerator(object):

    def __init__(self):
        pass

    def __call__(
            self, nengo_network, machine_time_step,
            nengo_random_number_generator_seed, decoder_cache,
            utilise_extra_core_for_output_types_probe,
            nengo_nodes_as_function_of_time,
            function_of_time_nodes_time_period, insert_interposers,
            print_timings, do_timings, xml_paths,
            pacman_executor_provenance_path,
            nengo_ensemble_profile, nengo_ensemble_profile_num_samples,
            receive_buffer_port, receive_buffer_host, minimum_buffer_sdram,
            maximum_sdram_for_sink_vertex_buffing,
            using_auto_pause_and_resume, time_between_requests,
            buffer_size_before_receive, spike_buffer_max_size,
            variable_buffer_max_size, machine_time_step_in_seconds):

        # create data holders
        inputs = dict()
        algorithms = list()
        outputs = list()
        tokens = list()
        required_tokens = list()
        optional_algorithms = list()

        # add nengo_spinnaker_gfe algorithms
        algorithms.append("NengoApplicationGraphBuilder")
        if insert_interposers:
            algorithms.append("NengoUtiliseInterposers")

        # add nengo_spinnaker_gfe inputs
        inputs["NengoModel"] = nengo_network
        inputs["MachineTimeStep"] = machine_time_step
        inputs["NengoRandomNumberGeneratorSeed"] = (
            nengo_random_number_generator_seed)
        inputs["NengoDecoderCache"] = decoder_cache
        inputs["NengoUtiliseExtraCoreForProbes"] = (
            utilise_extra_core_for_output_types_probe)
        inputs["NengoNodesAsFunctionOfTime"] = nengo_nodes_as_function_of_time
        inputs["NengoNodesAsFunctionOfTimeTimePeriod"] = (
            function_of_time_nodes_time_period)
        inputs["NengoEnsembleProfile"] = nengo_ensemble_profile
        inputs["NengoEnsembleProfileNumSamples"] = (
            nengo_ensemble_profile_num_samples)
        inputs["ReceiveBufferPort"] = receive_buffer_port
        inputs["ReceiveBufferHost"] = receive_buffer_host
        inputs["MinBufferSize"] = minimum_buffer_sdram
        inputs["MaxSinkBuffingSize"] = maximum_sdram_for_sink_vertex_buffing
        inputs["UsingAutoPauseAndResume"] = using_auto_pause_and_resume
        inputs["TimeBetweenRequests"] = time_between_requests
        inputs["BufferSizeBeforeReceive"] = buffer_size_before_receive
        inputs["SpikeBufferMaxSize"] = spike_buffer_max_size
        inputs["VariableBufferMaxSize"] = variable_buffer_max_size
        inputs["MachineTimeStepInSeconds"] = machine_time_step_in_seconds

        # Execute the algorithms
        executor = PACMANAlgorithmExecutor(
            algorithms=algorithms, optional_algorithms=optional_algorithms,
            inputs=inputs, tokens=tokens,
            required_output_tokens=required_tokens, xml_paths=xml_paths,
            required_outputs=outputs, do_timings=do_timings,
            print_timings=print_timings,
            provenance_name="nengo_graph_to_application_graph",
            provenance_path=pacman_executor_provenance_path)
        executor.execute_mapping()

        return (executor.get_item("NengoOperatorGraph"),
                executor.get_item("NengoHostGraph"),
                executor.get_item("NengoGraphToAppGraphMap"),
                executor.get_item("AppGraphToNengoOperatorMap"),
                executor.get_item("NengoRandomNumberGenerator"))
