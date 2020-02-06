PICK_PLACE_METHOD  = 'pickpush'
# SECTOR_SIZE = 5
SECTOR_SIZE = 8
BASE_PC_BOUNDS = [(-.17,  .15),
		  ( .17,  .15),
		  ( .17, -.15),
		  (-.17, -.15)]

BASE_HEIGHT_BOUNDS = (.39, .49)

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
PC_DENSITY = 5000            # further reduction

DISPLAY_IMG_KEYPOINT = True
DISPLAY_PC_RGB       = True
DISPLAY_PC_DEPTH_MAP = True
DISPLAY_PC_GRASPS    = True
DISPLAY_PC_KEYPOINT  = True
DISPLAY_PC_CLUSTERS  = True

IMG_KEYPOINT_TOPIC   = '/keypoints'
PC_DEPTH_MAP_TOPIC   = '/pc_depth_map'
PC_RGB_TOPIC         = '/pc_rgb'
PC_GRASPS_TOPIC      = '/pc_grasps'
PC_KP_TOPIC          = '/pc_kp'
PC_CLUSTER_TOPIC     = '/pc_clusters'

CLUSTER_MIN_SZ    = 20
# CLUSTER_EPS       = 1  # computed, in thousands...
# CLUSTER_EPS_DELTA = 1

INCH             = 0.0254
GRIPPER_WIDTH    = 0.9 * INCH
GRIPPER_LEN      = 1.0 * INCH
GRIPPER_HEIGHT   = 1.0 * INCH
MIN_GRIP_HEIGHT  = 0.25 * INCH
# MIN_OBJ_HEIGHT   = 0.25 * INCH
MIN_OBJ_HEIGHT   = 0.30 * INCH
GRIP_EVAL_RADIUS = 1.5 * INCH / 2
GRIPPER_OFFSET   = 1.3 * INCH

DEG20_IN_RADIANS = 0.349066

COMPUTE_KEYPOINT   = True
FAVOR_KEYPOINT     = False
KP_OOB_THRESH = 5   
OOB_THRESH    = 0
MIN_NEIGHBOR_THRESH = 5

IMG_WIDTH  = 640
IMG_HEIGHT = 480

# keypoint magic numbers via experimentation
# crop: (first_row, last_row)(first col, last col)
KP_IMG_CROP_DIM = [(115, IMG_HEIGHT), (0, 590)]  
# margin: (top_left, top_right)(bottom_left, bottom_right)
KP_IMG_MARGIN_DIM = [(119, 518), (40,590)] 
# map: (margins)(ratios)
KP_IMG_PC_MAP = [(110, 90), (1.3, 1.03)]

HISTORY_NUM_EVENTS = 5

BIGNUM = 100000000
