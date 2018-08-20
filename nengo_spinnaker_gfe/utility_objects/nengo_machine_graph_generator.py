from pacman.executor import PACMANAlgorithmExecutor


class NengoMachineGraphGenerator(object):

    def __init__(self):
        pass

    def __call__(
            self, system_pre_allocated_resources_inputs, max_machine_outputs,
            max_machine_available, steps, partitioning_algorithm,
            system_pre_alloc_res_algorithms, print_timings, do_timings,
            xml_paths, pacman_executor_provenance_path, nengo_operator_graph,
            machine_time_step, minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_sink_vertex_buffing, using_auto_pause_and_resume,
            receive_buffer_port, time_between_requests,
            buffer_size_before_receive):

        inputs = dict()
        algorithms = list()

        # update inputs with the system pre allocated inputs
        inputs.update(system_pre_allocated_resources_inputs)
        inputs.update(max_machine_outputs)

        # add partitioning inputs
        inputs["MemoryVirtualMachine"] = max_machine_available
        inputs["TotalMachineTimeSteps"] = steps
        inputs["NengoOperatorGraph"] = nengo_operator_graph
        inputs["MachineTimeStep"] = machine_time_step
        inputs["MinBufferSize"] = minimum_buffer_sdram
        inputs["MaxSinkBuffingSize"] = maximum_sdram_for_sink_vertex_buffing
        inputs["UsingAutoPauseAndResume"] = using_auto_pause_and_resume
        inputs["ReceiveBufferHost"] = receive_buffer_host
        inputs["ReceiveBufferPort"] = receive_buffer_port
        inputs["TimeBetweenRequests"] = time_between_requests
        inputs["BufferSizeBeforeReceive"] = buffer_size_before_receive

        # update algorithms with system pre allocated algor's
        algorithms.extend(system_pre_alloc_res_algorithms)
        algorithms.append(partitioning_algorithm)

        outputs = ["MemoryMachineGraph", "NengoGraphMapper"]

        # Execute the algorithms
        executor = PACMANAlgorithmExecutor(
            algorithms=algorithms, optional_algorithms=[],
            inputs=inputs,  tokens=[],
            required_output_tokens=[],
            required_outputs=outputs, print_timings=print_timings,
            do_timings=do_timings, xml_paths=xml_paths,
            provenance_name="nengo_application_graph_to_machine_graph",
            provenance_path=pacman_executor_provenance_path)
        executor.execute_mapping()

        # update spinnaker with app graph
        return (executor.get_item("MemoryMachineGraph"),
                executor.get_item("NengoGraphMapper"))
