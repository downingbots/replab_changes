PICK_PLACE_METHOD  = 'pickpush'
# SECTOR_SIZE = 5
SECTOR_SIZE = 4
BASE_PC_BOUNDS = [(-.17,  .15),
		  ( .17,  .15),
		  ( .17, -.15),
		  (-.17, -.15)]

BASE_HEIGHT_BOUNDS = (.39, .55)

# Playground is an area of ATTENTION
PLAYGROUND     = [(-.08,  -.08),
		  ( .0,   .08)]

# use rgb derived from transformed pc 
# pickpush policy is based purely on this rgbd
# rgb camera is: 480x640 (307200 pixels)
# 68x60 imh produces  8160 ~.2 inches pixels)
# 136 x 120 produces 16320 ~.1 inch pixels
# 272 x 240 produces 65280 ~.05 inch pixels
RGB_DEPTH_FROM_PC = True # False is original code
RGB_WIDTH  = 136
RGB_HEIGHT = 120
RGB_PC_DENSITY = 65280       # 4 to 1 ratio before reduction
PC_DENSITY = 15000           # further reduction

# The following identified all clusters tested without over-filtering.
# However, there are far more false positives (mostly with the base.)
# Try "pushes" over these low base clusters.
# Currently, tune before/after images to ensure all clusters are
# identified with few false positives
# Possibly could start with over-filtering a scene and later under-filtering
# a scene (HIGH_FILTER, LOW_FILTER)
# OCTOMAP_WARMUP       = 10    # number of octomap images before processing
OCTOMAP_WARMUP       = 8    # number of octomap images before processing
NUM_FILTER_SEGMENT   = 2    # number of additional plane segmentations to perform (min 0)

FILTER_BASE          = True  
# FILTER_BASE          = False  
CLUSTER_MIN_SZ       = 10

DISPLAY_PC_RGB       = True
DISPLAY_PC_GRASPS    = True
DISPLAY_PC_CLUSTERS  = True

OBB_TOPIC            = '/obb'
PC_RGB_TOPIC         = '/pc_rgbd'
PC_GRASPS_TOPIC      = '/pc_grasps'
PC_CLUSTER_TOPIC     = '/pc_clusters'
PC_OCTOMAP_TOPIC     = '/pc_octomap'              
PC_OCTOCLUSTER_TOPIC = '/pc_octocluster'      
PC_OCTOBASE_TOPIC    = '/pc_octobase'
OCTOMAP_TOPIC        = '/octomap_point_cloud_centers' # from octomap server

# BASE_PLANE_MIN_PCT   = 20     # CURRENTLY UNUSED

INCH             = 0.0254
GRIPPER_WIDTH    = 0.9 * INCH
GRIPPER_LEN      = 1.0 * INCH
GRIPPER_HEIGHT   = 1.0 * INCH
MIN_GRIP_HEIGHT  = 0.25 * INCH
MIN_OBJ_HEIGHT   = 0.9 * INCH
# MIN_OBJ_HEIGHT   = 0.2 * INCH
# MIN_OBJ_HEIGHT   = 0.25 * INCH
# MIN_OBJ_HEIGHT   = 0.30 * INCH
# MIN_OBJ_HEIGHT   = 0
# GRIP_EVAL_RADIUS = 1.5 * INCH / 2
GRIP_EVAL_RADIUS = 1.5 * INCH 
GRIPPER_OFFSET   = 1.3 * INCH

OCTOMAP_RESOLUTION = .1 * INCH
# OCTOMAP_RESOLUTION = .003682    # octomap resolution for 5000 pc points

DEG20_IN_RADIANS = 0.349066

COMPUTE_KEYPOINT   = False
FAVOR_KEYPOINT     = False
KP_OOB_THRESH = 5   
OOB_THRESH    = 0
MIN_NEIGHBOR_THRESH = 5

IMG_WIDTH  = 640
IMG_HEIGHT = 480

HISTORY_NUM_EVENTS = 5

BIGNUM = 100000000
ERROR_MARGIN = .1

WORLD_HISTORY_LEN = 5

CALIBRATION_METHODS = ('none', 'manual', 'auto')
