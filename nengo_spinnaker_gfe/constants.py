from enum import Enum

# enums associated with the ports used by the nengo_spinnaker_gfe tools
OUTPUT_PORT = Enum(
    value="OUTPUT_PORT",
    names=[('STANDARD', 0)])

INPUT_PORT = Enum(
    value="INPUT_PORT",
    names=[('STANDARD', 0)])

ENSEMBLE_OUTPUT_PORT = Enum(
    value="ENSEMBLE_OUTPUT_PORT",
    names=[('NEURONS', 0),
           ('LEARNT', 1)])

ENSEMBLE_INPUT_PORT = Enum(
    value="ENSEMBLE_INPUT_PORT",
    names=[('NEURONS', 0),
           ('GLOBAL_INHIBITION', 1),
           ('LEARNT', 2),
           ('LEARNING_RULE', 3)])

KEY_FIELDS = Enum(
    value="KEY_FIELDS",
    names=[('CLUSTER', 0)]
)

MATRIX_CONVERSION_PARTITIONING = Enum(
    value="MATRIX_CONVERSION_PARTITIONING",
    names=[('ROWS', 0),
           ('COLUMNS', 1)])

USER_FIELDS = Enum(
    value="KEY_SPACE_USER_FIELD",
    names=[("NENGO", 0),
           ("NOT_NENGO", 1)])

# key allocation fields
USER_FIELD_ID = "user"
CONNECTION_FIELD_ID = "connection_id"
CLUSTER_FIELD_ID = "cluster"
INDEX_FIELD_ID = "index"
INDEX_FIELD_START_POINT = 0

# key allocation tags
ROUTING_TAG = "routing"
FILTER_ROUTING_TAG = "filter_routing"

# key field size in bits
KEY_BIT_SIZE = 32

# math constants
CONVERT_MILLISECONDS_TO_SECONDS = 1000
SECONDS_TO_MICRO_SECONDS_CONVERTER = 1e6

# default timings
DEFAULT_DT = 0.001
DEFAULT_TIME_SCALE = 1.0

# DSG memory calculation
BYTE_TO_WORD_MULTIPLIER = 4
WORD_TO_BIT_CONVERSION = 32
BYTES_PER_KEY = 4

# sizes of elements for filter dsg regions
N_FILTER_TYPES = 3
N_FILTER_COUNT_SIZE = 1

# sizes of elements for routing dsg regions
ROUTING_N_ROUTES_SIZE = 1
ROUTING_ENTRIES_PER_ROUTE = 4

# flag constants used around the codebase
DECODERS_FLAG = "decoders"
DECODER_OUTPUT_FLAG = "decoded_output"
RECORD_OUTPUT_FLAG = "output"
RECORD_SPIKES_FLAG = "spikes"
RECORD_VOLTAGE_FLAG = "voltage"
ENCODERS_FLAG = "encoders"
SCALED_ENCODERS_FLAG = "scaled_encoders"

# graph constants
APP_GRAPH_NAME = "nengo_operator_graph"
INTER_APP_GRAPH_NAME = "nengo_operator_graph_par_way_interposers"
MACHINE_GRAPH_LABEL = "machine graph"


# sdp ports used by c code, to track with fec sdp ports.
SDP_PORTS = Enum(
    value="SDP_PORTS_READ_BY_C_CODE",
    names=[("SDP_RECEIVER", 6)])
