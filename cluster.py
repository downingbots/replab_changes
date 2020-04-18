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
      self.cluster['centroid'] = centroid          # current centroid 
      self.cluster['shape'] = shape                # normalized consolidated PC
      self.cluster['obb'] = obb                    # 3 sets of (x,y,z)
      self.cluster['obb_attr'] = obb_attr          # 3 sets of (x,y,z)
      self.cluster['shape_attr'] = shape_attr      # ["name",value] pair
                                                   # bounding box
                                              # shape_type: cyl, box, sphere...
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
      self.cluster['state'] = state                # e.g., ISOLATED 
                                              # rotated, flipped, pushed
                                              # flat spots, rolls, container
                                              # soft
      self.cluster['interaction'] = interaction    # with cluster_id list
      self.cluster['interaction_attr'] = interaction_attr  # ["name",value] pair
      # pointer to self.cluster not a deep copy!
      self.cluster['normalized_shape'] = []

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
    # part of WorldState?
    def find_attr(self, attr, key):
      # key = key_value[0]
      attr = self.cluster[attr]
      if (attr != None and len(attr) > 0 and attr[0] == key):
          return attr         # [key, value1 ...]
      return None

    def set_attr(self, attr, key, key_value):
      key = key_value[0]
      attr = self.cluster[attr]
      if (attr[0] == key):
          attr[1] = key_value
          # self.clusters[c_id][attr][i] = key_value
          return True
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
