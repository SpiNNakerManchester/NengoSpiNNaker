import math
import numpy

from collections import defaultdict

from nengo_spinnaker_gfe.machine_vertices.interposer_machine_vertex import \
    InterposerMachineVertex
from nengo_spinnaker_gfe.nengo_exceptions import \
    NotValidOutgoingPartitionIdentifier
from nengo_spinnaker_gfe.nengo_filters.\
    filter_and_routing_region_generator import \
    FilterAndRoutingRegionGenerator
from nengo_spinnaker_gfe.overridden_mapping_algorithms.\
    nengo_partitioner import NengoPartitioner
from pacman.executor.injection_decorator import inject
from pacman.model.graphs.common import Slice
from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex
from nengo_spinnaker_gfe import constants, helpful_functions


class InterposerApplicationVertex(AbstractNengoApplicationVertex):
    """Operator which receives values, performs filtering, applies a linear
        transform and then forwards the resultant vector(s).

        The input and output vector(s) may be sufficiently large that the load
        related to receiving all the packets, filtering the input vector, 
        applying
        the linear transform and transmitting the resultant values may be beyond
        the computational or communication capabilities of a single chip or
         core.
        The output vectors can be treated as a single large vector which is 
        split
        into smaller vectors by transmitting each component with an appropriate
        key; hence we can consider the entire operation of the filter component 
        as
        computing:

        ..math:: c[z] = \mathbf{A} b[z]

        Where **A** is the linear transform applied by the filter operator,
        :math:`b[z]` is the filtered input vector and :math:`c[z]` is the 
        nominal
        output vector.

        If **A** is of size :math:`m \times n` then *n* determines how many 
        packets
        each processing core (or group of processing cores) needs to receive and
        *m* determines how many packets each processing core (or group of cores)
        needs to transmit. To keep the number of packets received small we 
        perform
        column-wise partition of A such that:

        ..math:: c[z] = \mathbf{A_1} b_1[z] + \mathbf{A_2} b_2[z]

        Where :math:`\mathbf{A_x} b_x[z]` is the product computed by one set of
        processing cores and :math:`c[z]` is the resultant vector as 
        constructed by
        any cores which receive packets from cores implementing the filter
        operator. Note that the sum of products is computed by the receiving
         cores.
        **A** and `b` are now partitioned such that **A** is of size :math:`m
        \times (\frac{n}{2})` and `b` is of size :math:`\frac{n}{2}`; this 
        reduces
        the number of packets that need to be received by any group of cores
        implementing the filter operator.

        To reduce the number of packets which need to be transmitted by each 
        core
        we partition :math:`A_x` into rows such that:

        ..math::
            c =
            \begin{pmatrix}
              A_{1,1}~b_1 & + & A_{1,2}~b_2\\
              A_{2,1}~b_1 & + & A_{2,2}~b_2
            \end{pmatrix}

        Where, in this example, :math:`A_{x,y}` is of size :math:`\frac{m}{2}
        \times \frac{n}{2}`. Hence both the number of packets received and
        transmitted by each core has been halved, and the number of
        multiply-accumulates performed by each core has been quartered.  This
        reduction in communication and computation in the filter operator is
        achieved at the cost of requiring any operator with input `c` to 
        receive
        twice as many packets as previously (one set of packets for each
        column-wise division) and to perform some additions.
        """

    __slots__ = [
        # ?????
        "_size_in",

        #
        "_groups"
        ]

    # Maximum number of columns and rows which may be
    # handled by a single processing core. The defaults (128 and 64
    # respectively) result in the overall connection matrix being
    # decomposed such that (a) blocks are sufficiently small to be stored
    # in DTCM, (b) network traffic is reduced.

    # NB: max_rows and max_cols determined by experimentation by AM and
    # some modelling by SBF.
    # Create as many groups as necessary to keep the size in of any group
    # less than max_cols.

    MAX_COLUMNS_SUPPORTED = 128
    MAX_ROWS_SUPPORTED = 64

    n_interposer_machine_vertices = 0

    def __init__(self, size_in, label, rng, seed):
        """Create a new parallel Filter.
        
        :param size_in:  Width of the filter (length of any incoming signals).
        :type size_in: int
        :param rng: the random number generator for generating seeds
        :param label: the human readable label
        :type label: str
        :param seed: random number generator seed
        :type seed: int
        """
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._size_in = size_in

    @property
    def size_in(self):
        return self._size_in

    @property
    def groups(self):
        return self._groups

    @inject({
        "machine_time_step": "MachineTimeStep",
        "operator_graph": "NengoOperatorGraph"})
    @overrides(AbstractNengoApplicationVertex.create_machine_vertices,
               additional_arguments=["machine_time_step",
                                     "operator_graph"])
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper,
            machine_time_step, operator_graph):
        """Partition the transform matrix into groups of rows and assign each
        group of rows to a core for computation.
    
        If the group needs to be split over multiple chips (i.e., the group is
        larger than 17 cores) then partition the matrix such that any used
        chips are used in their entirety.
        """

        # verify that channel output port standard exists. If not no vertices
        #  will be made
        outgoing_partitions = operator_graph\
            .get_outgoing_edge_partitions_starting_at_vertex(self)
        output_standard_partitions = list()
        for outgoing_partition in outgoing_partitions:
            if (outgoing_partition.identifier.source_port ==
                    constants.OUTPUT_PORT.STANDARD):
                output_standard_partitions.append(outgoing_partition)
            else:
                raise NotValidOutgoingPartitionIdentifier(
                    "The outgoing partitions are expected to be of type {}. "
                    "The outgoing partition {} was not.".format(
                        constants.OUTPUT_PORT.STANDARD, outgoing_partition))

        # TODO IS THIS A VALID APPROACH!?
        if len(output_standard_partitions) == 0:
            return

        # locate the incoming standard partitions
        filter_keys = 0
        filters = defaultdict(list)

        for incoming_edge in operator_graph.get_edges_ending_at_vertex(self):
            if incoming_edge.input_port == constants.INPUT_PORT.STANDARD:
                FilterAndRoutingRegionGenerator.add_filters(
                    filters, incoming_edge,
                    operator_graph.get_outgoing_partition_for_edge(
                        incoming_edge), minimise=True, width=self._size_in)
            filter_keys += 1

        # there are standard outgoings, so build machine verts
        n_groups = int(math.ceil(self._size_in // self.MAX_COLUMNS_SUPPORTED))
        filter_slices = NengoPartitioner.divide_slice(
            Slice(0, self._size_in), n_groups)

        for filter_slice in filter_slices:
            self._generate_cluster(
                filter_slice, filter_keys, filters, output_standard_partitions,
                machine_graph, graph_mapper, resource_tracker,
                machine_time_step)

    def _generate_cluster(
            self, filter_slice, filter_keys, filters,
            output_standard_partitions, machine_graph, graph_mapper,
            resource_tracker, machine_time_step):

        # Get the output transform, keys and slices for this slice of the
        # filter.
        transform, n_keys, output_slices = self._get_transforms_and_keys(
            output_standard_partitions, filter_slice)

        size_out = transform.shape[0]

        # Build as many vertices as required to keep the number of rows
        # handled by each core below max_rows.
        for output_slice in NengoPartitioner.divide_slice(
                initial_slice=Slice(0, size_out),
                n_slices=int(math.ceil(size_out // self.MAX_ROWS_SUPPORTED))):

            # Build the transform region for these cores
            transform_data = \
                helpful_functions.convert_matrix_to_machine_vertex_level(
                    transform, output_slice,
                    constants.MATRIX_CONVERSION_PARTITIONING.ROWS)

            # build machine vertex
            machine_vertex = InterposerMachineVertex(
                filter_slice, output_slice, transform_data, n_keys, filter_keys,
                output_slices, machine_time_step, filters,
                "interposer_with-slice{}:{}_for_interposer{}".format(
                    filter_slice.lo_atom, filter_slice.hi_atom, self),
                self.constraints)

            # update graph objects
            machine_graph.add_vertex(machine_vertex)
            graph_mapper.add_vertex_mapping(
                machine_vertex=machine_vertex, application_vertex=self)
            resource_tracker.allocate_resources(
                machine_vertex.resources_required)

            InterposerApplicationVertex.n_interposer_machine_vertices += 1
        return self.cores

    def add_constraint(self, constraint):
        self._constraints.add(constraint)

    @property
    def constraints(self):
        return self._constraints

    @staticmethod
    def _get_transforms_and_keys(outgoing_partitions, columns):
        """Get a combined transform matrix and a list of keys to use to \
        transmit elements transformed with the matrix.  This method also \
        returns a list of signal parameters, transmission parameters and the \
        slice of the final transform matrix that they are associated with.
        """
        transforms = list()
        n_keys = 0
        slices = list()

        start = end = 0
        for outgoing_partition in outgoing_partitions:
            # Extract the transform
            transmission_parameter = \
                outgoing_partition.identifier.transmission_parameter
            transform = transmission_parameter.full_transform(
                slice_in=False, slice_out=False)[:, columns]

            if outgoing_partition.identifier.latching_required:
                # If the signal is latching then we use the transform exactly
                # as it is.
                keep = numpy.array([True for _ in range(transform.shape[0])])
            else:
                # If the signal isn't latching then we remove rows which would
                # result in zero packets.
                keep = numpy.any(transform != 0.0, axis=1)

            transforms.append(transform[keep])
            end += transforms[-1].shape[0]

            slices.append((transmission_parameter, set(range(start, end))))
            start = end

            for i, k in zip(range(transform.shape[0]), keep):
                if k:
                    n_keys += 1

        # Combine all the transforms
        if len(transforms) > 0:
            transform = numpy.vstack(transforms)
        else:
            transform = numpy.array([[]])
        return transform, n_keys, slices
