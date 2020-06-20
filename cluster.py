#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
from sensor_msgs.msg import (Image)
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

from utils_grasp import *
from config_grasp import *
from keypoint import *
from obb import *
import utils_grasp
import cv2 as cv
from math import sqrt
import scipy.stats

class ClusterState:

    ################################
    # TODO: make into classes / superclasses with inheritance
    # base class
    # latest pointcloud img class
    # persistent world class
    ################################
    # BOTH APIs : World and Latest PC
    ################################
    def __init__(self):
      self.state = 'UNINITIALIZED' 
      self.cluster = {}

    def create_cluster(self, id = None, status = None, centroid = None, 
               shape = None, shape_attr = None, obb = None, obb_attr = None,
               num_locations = None, location = None, 
               num_grasps = None, grasps = None, grasp_attr = None,
               interaction = None, interaction_attr = None,
               state = None): 

      # self.cluster = {}  # done during init
      self.cluster['id'] = id
      self.cluster['status'] = status # known, possible match, possible_combined
                                    # obsolete, removed
                 # ACTIVE, INACTIVE
      self.cluster['centroid'] = centroid          # current centroid 
      self.cluster['shape'] = shape                # normalized consolidated PC
      self.cluster['obb'] = obb                    # 3 sets of (x,y,z)
      self.cluster['obb_attr'] = obb_attr          # 3 sets of (x,y,z)
      self.cluster['shape_attr'] = shape_attr      # ["name",value] pair
                                                   # bounding box
                                              # shape_type: cyl, box, sphere...
      self.cluster_move_history = {}
      self.cluster_grasp_history = {}
      if num_locations == None:
        self.cluster['num_locations'] = 0            
        self.cluster['location'] = []              # a history of locations
      if num_grasps == None:
        self.cluster['num_grasps'] = 0
        self.cluster['grasps'] = []              
        self.cluster['grasp_attr'] = []            # ["name",value] pair
      if location is not None:
        self.cluster['num_locations'] += 1
        self.cluster['location'].append(location)  # center=x,y,z + orientation
      self.cluster['state'] = state                
				      # e.g., ISOLATED 
				      # rotated, flipped, pushed
				      # flat spots, rolls, container
				      # soft
      # interaction is for entropy. Drop ontop of object, place on top, push into..`
      self.cluster['interaction'] = interaction    # with cluster_id list
      self.cluster['interaction_attr'] = interaction_attr  # ["name",value] pair
      # pointer to self.cluster not a deep copy!
      self.cluster['normalized_shape'] = []
      self.cluster['status'] = "INACTIVE"     # ACTIVE, INACTIVE, UNKNOWN,
      self.cluster['type'] = "EMPTY"      # OBJECT, MULTI_OBJECT, BASE, EMPTY
      self.cluster['relation'] = []
      # [None, 'PART_OF', 'CONSISTS_OF', 'CONTAINED_BY', 'CONTAINS', 'STACKED_ON', 'STACKED_BELOW']
      self.cluster['shape_type'] = None   # CONTAINER, ROLLABLE, FLAT_SURFACE


    #####################
    # BOTH UTILITIES
    #####################
    def compute_bounding_box(self):
      pc_cluster_shape = self.cluster['shape']
      # points = [[p[0], p[1], p[2]] for p in pc_cluster.shape]
      points = [[p[0], p[1], p[2]] for p in pc_cluster_shape]
      min = [BIGNUM, BIGNUM, BIGNUM]
      max = [-BIGNUM, -BIGNUM, -BIGNUM]
      for p in pc_cluster_shape:
        for i in range(3):
          if min[i] > p[i]:
            min[i] = p[i]
          if max[i] < p[i]:
            max[i] = p[i]
      # print(self.cluster['id']," cluster numpts:",len(pc_cluster_shape))
      obb = OBB.build_from_points(points)
      # print("obbTmin:",obb.get_min)
      # print("obbTmax:",obb.get_max)
      self.cluster['obb'] = obb

    # ARD TODO: use self.clusters[c_id]
    def in_bounding_box(self, point):
      # from scipy.spatial import ConvexHull

      bb = self.cluster['obb'] 
      if bb is None:
        print("no bounding box ")
        # return None
        return False
      # print("bounding box :",bb.points)
      # print("bounding box2 :",bb.points3d)
      return OBB.in_obb(bb, point)


    #####################
    # LATEST PC UTILITIES
    #####################
    # ARD TODO: be consistent with self.clusters[c_id]
    def normalize(self):
      self.cluster['normalized_centroid'] = np.mean(self.cluster['shape'], axis=0)
      # print("center ", self.cluster['center'], " centroid ", self.cluster['centroid'])
      for i in range(len(self.cluster['shape'])):
        self.cluster['normalized_shape'].append( self.cluster['shape'][i] - self.cluster['normalized_centroid'])

    ####################
    # WORLD UTILITIES
    ####################
    # cluster_move_history should be in cluster.py
    # Moved to goal.py
    def previously_tried_grasp(self, grasp):
      for prev_grasp in reversed(self.cluster_grasp_history):
        [action, result] = prev_grasp 
        if action[0] != "GRASP":
          continue
        prev_grasp = action[1]
        if distance_3d(grasp, prev_grasp) <= .25 * INCH and result == False:
          print("previously tried grasp:", action, result)
          return True
      return False

    def add_to_grasp_history(self, action, result):
      print("add_to_history", action, result)
      self.cluster_grasp_history.append([action,result])

    def add_to_history(self, action, result):
      print("add_to_history", action, result)
      self.cluster_move_history.append([action,result])

    # part of WorldState?
    def find_attr(self, attr, key):
      # key = key_value[0]
      attr = self.cluster[attr]
      if (attr != None and len(attr) > 0 and attr[0] == key):
          if key != None:
            return attr[key] 
          return attr         # [key, value1 ...]
      return None

    def set_attr(self, attr, key, key_value):
      attr = self.cluster[attr]
      if attr == None:
        return False
      cluster[attr][key] = key_value
      return False

    # for persistent world cluster
    def set_mean_stddev(self, attr, key):
      [key2, sum, sum2, n] = self.find_attr(attr,key)
      sum  += x
      sum2 += x*x
      n    += 1.0
      self.set_attr(attr,[key, sum, sum2, n])

    # for persistent world cluster
    def get_mean_stddev(self, attr, key):
      ret = self.find_attr(attr,key)
      if ret == None:
        return None, None
      [sum, sum2, n] = ret
      return (sum/n), sqrt(sum2/n - sum*sum/n/n)

    def grasp_in_playground(self):
      self.cluster['centroid'] = centroid          # current centroid 
      self.cluster['obb'] = bounding_box  # 3 sets of (x,y,z)
      # ARD: TODO
      pass

    #########################################
    # UTILITIES FOR INTEGRATING LATEST PC INTO WORLD 
    #########################################
    # part of WorldState?
    def compare_bounding_box(self, c_id, pc_cluster):
      # LOCATON DEPENDENT PROBABILITY
      # centroid changes when moved
      # yet, stddev still can be computed?
      cent_mean, cent_stddev = self.get_mean_stddev(c_id,'obb_attr',"CENTROID_STATS") 
      cent_val = pc_cluster['obb'].centroid
      # if unmoved, computed mean and variance centroid 
      for c_v in zip(c1, c2):
        cent_prob = scipy.stats.norm(cent_mean, cent_stddev).pdf(c_v)
      # rotation changes when moved
      # if unmoved, computed mean and stddev centroid 
      rot_mean, rot_stddev = self.get_mean_stddev(c_id,'obb_attr',"ROTATION_STATS") 
      rot_val = pc_cluster['obb'].rotation
      for r_v in zip(rot_val):
        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(r_v)
      loc_dep_prob = 1
      for i in range(3):
        loc_dep_prob = min(loc_dep_prob, rot_prob[i], cent_prob[i])
      #
      # LOCATON INDEPENDENT PROBABILITY
      # max & min changes over time, but max-min typically doesn't
      # 1 stdev -> 68%; 2 stdev -> 95%; 3 stdev -> 99.7%
      # pdf = probability distribution function
      c1min = self.clusters[c_id]['obb'].min
      c2min = pc_cluster['obb'].min
      c1max = self.clusters[c_id]['obb'].max
      c2max = pc_cluster['obb'].max
      #
      min_mean, min_stddev = self.get_mean_stddev(c_id,'obb_attr',"MIN_DIF_STATS") 
      max_mean, min_mean= self.get_mean_stddev(c_id,'obb_attr',"MAX_DIF_STATS") 
      for c1Xmax, c2Xmax in zip(c1max, c2max):
        for c1Xmin, c2Xmin in zip(c1min, c2min):
          # x, y, z
          val = abs(c1Xmax - c2Xmax)
          bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(val)
          val = abs(c1Xmin - c2Xmin)
          bb_max_prob = scipy.stats.norm(max_mean, max_stddev).pdf(val)
      loc_indep_prob = 1
      for i in range(3):
        loc_indep_prob = min(loc_indep_prob, bb_min_prob[i], bb_max_prob[i])
        loc_dep_prob = min(loc_dep_prob, loc_indep_prob)
      loc_dep_prob = loc_indep_prob * loc_dep_prob
      return loc_dep_prob, loc_indep_prob

    # associate grasp with the cluster
    def compare_location(self, c_id, pc_cluster):
      pass

    def compare_shapes(self, c_id, pc_cluster):
      # use bounding boxes instead
      pass

    # combine clusters A,B using T transform
    def combine_clusters(self, A, B, T):
      # translate points to their centroids
      centroid_A = np.mean(A, axis=0)
      centroid_B = np.mean(B, axis=0)
      AA = A - centroid_A
      BB = B - centroid_B
      AAT = np.dot(T, AA)
      C = np.concatenate([AAT, BB], axis=0)
      return C
  
    # select random pc subset
    def random_pc_subset(self, pc, num_pts):
      np.random.shuffle(pc)
      # ARD
      # need to ensure the keypoints are chosen;
      # exchange slots so that keypoints are on top
      if pc.shape[0] > num_pts:
          pc2 = pc[:num_pts]
      return pc2

    # icp wants same # points
    def make_same_num_points(A, B):
      if A.shape[0] > B.shape[0]:
        AA = random_pc_subset(A, B.shape[0])
        BB = B
      elif A.shape[0] < B.shape[0]:
        BB = random_pc_subset(B, A.shape[0])
        AA = A
      else:
        AA = A
        BB = B
      return AA,BB
  
    def obb_length_width(self):
      obb     = self.cluster['obb'] 
      pts = obb.points 
      # 4: leftmost, bottommost, farthest
      # 5: rightmost, bottommost, farthest
      # 6: rightmost, bottommost, closest
      # 7: leftmost, bottommost, closest
      len1 = distance_3d(pts[4],pts[5])
      len2 = distance_3d(pts[6],pts[7])
      len3 = distance_3d(pts[5],pts[6])
      len4 = distance_3d(pts[7],pts[4])
      s1 = min(len1, len2)
      s2 = min(len3, len4)
      if s1 < s2:
        return s2, s1
      return s1, s2
      
    def near_side(self):
      obb = self.cluster['obb'] 
      # use all the bottommost points #4-#7
      for pt_id,pt in enumerate(obb.points, 4):
        [pt, side, dist, spt0, spt1, spt2] = close_to_side(pt)
        # pt = check if point is close to side
        # spnt0 = up slope point on top/bot side, None for l/r sides
        # spnt1 = point on side
        # spnt2 = point a minimum distance from side
        if side != None and side != "no side":
          return True
      return False

    def plan_move_from_side(self):
      # the bounding box is located next to the side.
      # find the pont on the bounding box to start push at
      obb = self.cluster['obb'] 
      min_dist = BIGNUM
      retlst = []
      # use all the "closest" points #0 1 4 5
      idx_lst = [0,1,4,5]
      for idx_id,idx in enumerate(idx_lst):
        pt = obb.points[idx]
        ret = close_to_side(pt)
        [obb_pt, side, dist, pt0, pt1, pt2] = ret
        if pt1 != None and dist < min_dist:
          min_dist = dist
          retlst.append(ret)
      if len(retlst) == 0:
        print("None ret:", ret)
        return [None, None, None, None, None, None]
      if len(retlst) == 1:
        return retlst[0]
      else:
        min_dist = BIGNUM
        min_ret  = None
        corner_min_dist = BIGNUM
        corner_min_ret = None
        for ret in retlst:
          [obb_pt, side, dist, pt0, pt1, pt2] = ret
          if min_dist < dist:
            min_dist = dist
            min_ret = ret
          if side.startswith("corner:") and corner_min_dist < dist:
            corner_min_dist = dist
            corner_min_ret = ret
            corner.append(ret)
        if corner_min_ret != None:
          return corner_min_ret
        if len(retlst) == 2:
          pt0a = retlst[0][3]
          pt0b = retlst[1][3]
          if pt0a != None and pt0b != None:
            midpt = [(pt0a[i] + pt0b[i])/2 for i in range(3)]
            ret = close_to_side(midpt)
          elif pt0a != None:
            ret = retlst[0]
          else:
            ret = retlst[1]
          return ret
        return min_ret

    #################
    # TBD
    #################

    def analyze_grasp(self, pc):
      self.state = "ANALYZE_GRASP"
      # find cluster that was grabbed / rotated
      if DISPLAY_PC_CLUSTERS:
        self.pc_cluster_pub = rospy.Publisher(PC_CLUSTER_TOPIC, PointCloud2, queue_size=1)

    def analyze_lift(self, pc):
      self.state = "ANALYZE_LIFT"

    def analyze_drop(self, pc):
      self.state = "ANALYZE_DROP"

    # ?????
    def analyze_object(self):
      pass

    # world?
    def empty(self):
      pass

    def unchanged(self, cl1, cl2):
      pass

    # after drop, does it move?
    def moved(self):
      pass

    #########################
    # CLUSTER ATTRIBUTES
    # not required for initial implementation
    def rolls(self):
      pass

    def flat(self):
      # center
      pass

    def container(self):
      pass

    def cylinder(self):
      pass

    def sphere(self):
      pass


    #######
    def save_history(self):
      pass

    def cluster_contains(self, grasp):
      pass

    def rotate_angle(self):
      return DEG20_IN_RADIANS

    def get_action_mode(self):
      pass


    ###################
