PICK_PLACE_METHOD  = 'pickpush'
SECTOR_SIZE = 5
BASE_PC_BOUNDS = [(-.17,  .15),
		  ( .17,  .15),
		  ( .17, -.15),
		  (-.17, -.15)]

BASE_HEIGHT_BOUNDS = (.39, .49)

PC_DENSITY = 25000

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

