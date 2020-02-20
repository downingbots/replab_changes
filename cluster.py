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
               kp = None, kp_attr = None, full_kp_c_pc_map = None,
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
      # 'kp' is the kp derived from a single world cluster
      self.cluster['kp'] = kp                      # Keypoint class
      self.cluster['kp_attr'] = []                 # mean, stddev for distances 
      # kp_c_pc_mapping is for mapping the full-pc keypoint to a pc cluster
      self.cluster['kp_c_pc_mapping'] = full_kp_c_pc_map
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
      self.cluster['normalized_kp'] = []


    ###############################
    # LATEST PC APIs
    ###############################
#
#    def publish_pc(self):
#      if not DISPLAY_PC_CLUSTERS:
#        return
#      color = 4294967294 / 2
#      clusters_pc = []
#      for n in range(self.num_clusters):
#        color = color / 2
#        cluster_pc = [[p[0],p[1],p[2],color] for p_i, p in enumerate(self.cluster_PC[n])]
#        clusters_pc.append(cluster_pc)
#      cluster_pc = np.reshape(cluster_pc, (len(cluster_pc), 4))
#      fields = [PointField('x', 0, PointField.FLOAT32, 1),
#                PointField('y', 4, PointField.FLOAT32, 1),
#                PointField('z', 8, PointField.FLOAT32, 1),
#                # PointField('rgba', 12, PointField.UINT32, 1)]
#                PointField('rgb', 12, PointField.UINT32, 1)]
#      cluster_pc = point_cloud2.create_cloud(self.pc_header, fields, cluster_pc)
#      self.pc_cluster_pub.publish(cluster_pc)
#

    #####################
    # BOTH UTILITIES
    #####################
    # plane model segmentation
    def segment_cluster(self, pc):
      from sklearn import linear_model
      from sklearn.metrics import r2_score, mean_squared_error

      # seg.set_normal_distance_weight(0.1)
      # seg.set_method_type(pcl.SAC_RANSAC)
      # seg.set_max_iterations(100)
      # seg.set_distance_threshold(-1.03)

      points = pc
      max_iterations=100
      best_inliers = None
      n_inliers_to_stop = len(points)
      self.point = np.mean(points, axis=0)
      # data is an np.array
      # data_adjust = data - mean
      data_adjust = points - self.point
      matrix = np.cov(data_adjust.T)  # transpose data_adjust
      eigenvalues, self.normal = np.linalg.eig(matrix)
      n_best_inliers = 0
      # max_dist = 1e-4  # 1e-4 = 0.0001.
      # max_dist = .0025 # 0.1 inches
      # max_dist = .0025 # 0.1 inches
      # max_dist = .005 # 0.1 inches
      max_dist = .03
      print_once = True
      # max_dist = .001
      for i in range(max_iterations):
          # k_points = sampler.get_sample()
          normal = np.cross(points[1] - points[0], points[2] - points[0])
          self.point = points[0]
          if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            if print_once:
              print_once = False
              print("normal: ",normal)
            self.normal = [1,1,1]
          else:
            self.normal = normal / np.linalg.norm(normal)
          vectors = points - self.point
          all_distances = np.abs(np.dot(vectors, self.normal))
          inliers = all_distances <= max_dist
          n_inliers = np.sum(inliers)
          if n_inliers > n_best_inliers:
              n_best_inliers = n_inliers
              best_inliers = inliers
              if n_best_inliers > n_inliers_to_stop:
                  break
      # print("plane: ", best_inliers)            # true/false array
      # print("len plane: ", len(best_inliers))
      # print("len pc   : ", len(pc))             # same len as above
      # for i in range(len(best_inliers)):
      #   if i % 10 == 0:
      #     print(pc[i])
      return best_inliers

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
      # cluster_obb = OBB()
      # obb = cluster_obb.build_from_points(points)
      print(self.cluster['id']," cluster numpts:",len(pc_cluster_shape))
      obb = OBB.build_from_points(points)
      # print("c min  :",min)
      # print("c max  :",max)
      print("obbTmin:",obb.get_min)
      print("obbTmax:",obb.get_max)
      # print("obb min:",obb.min)
      # print("obb max:",obb.max)
      # print("POINTS",points)
      # print("obb cnt",obb.centroid)

      # print("number of cluster points: ",len(pc_cluster_shape), " centroid ", obb.centroid())
      # print("num cluster pnts: ",len(pc_cluster_shape), " min/max ", obb.min, obb.max)
      self.cluster['obb'] = obb
      # print("obb:", obb.get_min, obb.get_max)
      # obb return values:
      #   obb.centroid()
      #   obb.rotation    # u,v,w=> [0],[1],[2] transform matrix
      #   obb.min         # min[0], min[1], min[2]
      #   obb.max

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

#      hull = ConvexHull(bb.points)
#      # print("convex hull volume", hull.volume)
#
##      tolerance=1e-12
##      return all(
##              (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
##              for eq in hull.equations)
#
## [ point_in_hull(point, hull) for point in cloud ]
## [True, True, True]
#      # new_pts = bb.points + array(point)
#      # point should not be in the hull vertices
#      new_pts = array(point) + bb.points 
#      # print("new_pts:",new_pts)
#      new_hull = ConvexHull(new_pts)
#      # print("old hull: ", hull.vertices)      # if in obb: 1-8
#      # print("new hull: ", new_hull.vertices)  # 0-7
#      if len(hull.vertices) != len(new_hull.vertices):
#        print("Not in bounding box 1: ", len(hull.vertices),len(new_hull.vertices))
#        return False
#      for v in new_hull.vertices:
#        # if new_pts[v] not in bb.points:
#        for i in range(8):
#          if new_pts[v][0]!=bb.points[i][0] or new_pts[v][1]!=bb.points[i][1]:
#            print("Not in bounding box 2: ", new_pts[new_hull.vertices[v]])
#            return False
#      print("in bounding box")
#      return True

    #####################
    # LATEST PC UTILITIES
    #####################
    # ARD TODO: be consistent with self.clusters[c_id]
    def normalize(self):
      self.cluster['normalized_centroid'] = np.mean(self.cluster['shape'], axis=0)
      # print("center ", self.cluster['center'], " centroid ", self.cluster['centroid'])
      for i in range(len(self.cluster['shape'])):
        self.cluster['normalized_shape'].append( self.cluster['shape'][i] - self.cluster['normalized_centroid'])
      # ARD: normalize keypoint locations
      if self.cluster['kp'] != None:
        KPs = self.cluster['kp'].get_kp()
        if KPs != None:
          for i in range(len(KPs)):
            self.cluster['normalized_kp'].append(KPs[i] - self.cluster['normalized_centroid'])
      # mapping of pixel to x,y,z
      # kp = [[locx, locy, locz], loc2], descriptor][locations
      # return self.cluster

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



#
# Approach:
#   Initialize based upon first image:
#     Filter out tray from 3d image
#     Generate 2d image from 3d image
#     Compute and detect KPs from 2d image
#     Ensure KPs are in 3d image
#     Segment filtered 3d image into clusters
#     Based upon keypoints, store normalized keypoints into associated cluster
#     Store as a persistent stored cluster
#
#   Future image:
#     Filter out tray from 3d image
#     Generate 2d image from 3d image
#     Compute and detect KPs from 2d image
#     Segment filtered 3d image into clusters
#     Ensure KPs are in 3d image
#
#     For each world cluster, generate cluster 2d image
#     Compute and detect KPs for each cluster 2d image
#     Compare full set of new KPs to each world cluster KPs
#     Map new KPs to new clusters
#     Map new clusters to world clusters
#     compare and combine clusters
#     Future: handle rotation, flipping, rolling
#

#     Map new KPs to new clusters
#     Map new clusters to world clusters
#     compare and combine clusters
#     Future: handle rotation, flipping, rolling

