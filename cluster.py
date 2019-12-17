#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
import obb
from sensor_msgs.msg import (Image)
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

from utils_grasp import *
from config_grasp import *
import utils_grasp
import cv2 as cv
from math import sqrt

class ClusterState:

    def __init__(self, pc, KP):
      self.world_state = 'ISOLATE'   # ISOLATE, INTERACT
      self.num_clusters = 0
      self.cluster = []

    def create_cluster(self, id = None, status = None, centroid = None, 
               shape = None, shape_attr = None, bounding_box = None
               kp = None, kp_attr,
               num_locations = None, location = None, 
               num_grasps = None, grasps = None, grasp_attr = None,
               state = None): 
      cluster = {}
      cluster['id'] = id
      cluster['status'] = status    # known, possible match, possible_combined
                                    # obsolete, removed
      cluster['centroid'] = centroid          # current centroid 
      cluster['shape'] = shape                # normalized consolidated PC
      cluster['bounding_box'] = bounding_box  # 3 sets of (x,y,z)
      cluster['shape_attr'] = shape_attr      # ["name",value] pair
                                              # bounding box
                                              # shape_type: cyl, box, sphere...
      cluster['kp'] = kp                      # list of keypoints
      cluster['kp_attr'] = []                 # mean, stddev for distances 
      if num_locations == None:
        cluster['num_locations'] = 0            
        cluster['location'] = []              # a history of locations
      if num_grasps == None:
        cluster['num_grasps'] = 0
        cluster['grasps'] = []              
        cluster['grasp_attr'] = []            # ["name",value] pair
      if location is not None:
        cluster['num_locations'] += 1
        cluster['location'].append(location)  # center=x,y,z + orientation
      cluster['state'] = state                # e.g., ISOLATED 
                                              # rotated, flipped, pushed
                                              # flat spots, rolls, container
                                              # soft
      cluster['interaction'] = interaction    # with cluster_id list
      cluster['interaction_attr'] = interaction_attr  # ["name",value] pair

    def find_attr(self, c_id, field, key):
      for attr in self.cluster[c_id][field]:
        if (attr[0] == key):
          return attr         # [key, value1 ...]
      return None

    def set_attr(self, c_id, field, key_value):
      key = key_value[0]
      for i, attr in enumerate(self.cluster[c_id][field]):
        if (attr[0] == key):
          self.cluster[c_id][field][i] = key_value
          return True
      return False

    def find_cluster(self, id = None, center = None, shape = None, kp = None,
                     location = None, state = None):

    # store in file
    def save_world(self):

    # read from file
    def retore_world(self):

    def add_world_cluster(self, cluster)
      if cluster['id'] == None:
        cluster['id'] = self.num_clusters
        self.num_clusters += 1 
      self.world.append(cluster)

    def analyze_pc(self, pc, KP = None):
      db1 = DBSCAN(eps=.001, min_samples=3,
                   n_jobs=-1).fit(pc1)
      # Number of clusters in labels, ignoring noise if present.
      n1_clusters = len(set(db1.labels_)) - (1 if -1 in db1.labels_ else 0)
      print("DBSCAN: # ", n1_clusters)
      cluster = []
      for i in range(n1_clusters):
        cluster.append(self.create_cluster(id = i,kp = [], shape = [])
      if KP != None:
        kp = KP.get_kp() 

      for c in set(db1.labels):
        if c != -1:
          running_sum = np.array([0.0, 0.0, 0.0])
          counter = 0
          for i in range(pc.shape[0]):
              if db1.labels[i] == c:
                  running_sum += pc[i]
                  counter += 1
                  cluster[c]['shape'].append(pc[i])
                  if KP != None:
                    if [pc[i][0], pc[i][1]] in kp:
                      cluster[c]['kp'].append(pc[i])
                    print("kp found in cluster")
          center = running_sum / counter
          # ClusterState: [id, center, KPs, pc]
          cluster[c]['center] = center
          self.cluster_PCs.append(cluster_pc)

      for i, c in enumerate(cluster):
        # normalize shape
        cluster.shape = self.normalize(c)

      for i, pc_c in enumerate(cluster):
        # compute bounding box
        self.compute_bounding_box(c)
        for w_c_id, w_c in enumerate(self.cluster):
          # compare bounding box, location, keypoints, shape
          bb_prob, loc_indep_prob = self.compare_bounding_box(w_c_id, pc_c)
          # location included withing bounding box
          # loc_prob = self.compare_location(w_c_id, pc_c)
          kp_prob = self.compare_keypoints(w_c_id, pc_c)
          # shape_prob = self.compare_shapes(w_c_id, pc_c)
          total_prob = bb_prob*loc_prob*kp_prob
          np....[c,pc] = [bb_prob, loc_prob, kp_prob, total_prob]
          max_prob_clust[c] = [

        if max_prob_clust[c] > CLUSTER_MATCH:
          # ensure only 1-1 matches, may be multiple similar object

        if max_prob_clust[c] > CLUSTER_MATCH:
          self.combine_clusters(w_c_id,c)


       # multiple clusters with shared attributes?
       #    same shape (cube), color?

       for unmatched clusters:
          # same # of clusters?  how many unaccounted for?
          # combine bounding boxes of unaccounted in different ways
          did clusters get combined?
          did clusters get split?
          if location changed, did it just roll there?
          set probabilities?

          # compute a probability same
  
          # compute probability that 2 clusters were "joined" together
          #

    def normalize(self, cluster):
      cluster['centroid'] = np.mean(cluster['shape'], axis=0)
      cluster['shape']  = cluster['shape'] - cluster['centroid']

    def compute_bounding_box(self, cluster):
      import scipy.stats

      points = [p[0], p[1], p[2] for p in pc_cluster]
      obb = OBB.build_from_points(points)
      return obb
      # obb return values:
      #   obb.centroid()
      #   obb.rotation    # u,v,w=> [0],[1],[2] transform matrix
      #   obb.min         # min[0], min[1], min[2]
      #   obb.max

    def set_mean_stddev(cid, attr, key):
      [key2, sum, sum2, n] = self.find_attr(c_id,attr,key)
      sum  += x
      sum2 += x*x
      n    += 1.0
      self.set_attr(c_id,attr,[key, sum, sum2, n])

    def get_mean_stddev(cid, attr, key):
      [sum, sum2, n] = self.find_attr(c_id,attr,key)
      sum, sum2, n = sd.sum, sd.sum2, sd.n
      return (x/n), sqrt(sum2/n - sum*sum/n/n)

    def compare_bounding_box(self, c_id, pc_cluster):
      ###############################
      # LOCATON DEPENDENT PROBABILITY
      ###############################
      # centroid changes when moved
      # yet, stddev still can be computed?
      cent_mean, cent_stddev = self.get_mean_stddev(c_id,'boundingbox_attr',"CENTROID_STATS") 
      cent_val = pc_cluster['boundingbox'].centroid
      # if unmoved, computed mean and variance centroid 
      for c_v in zip(c1, c2):
        cent_prob = scipy.stats.norm(cent_mean, cent_stddev).pdf(c_v)
      # rotation changes when moved
      # if unmoved, computed mean and stddev centroid 
      rot_mean, rot_stddev = self.get_mean_stddev(c_id,'boundingbox_attr',"ROTATION_STATS") 
      rot_val = pc_cluster['boundingbox'].rotation
      for r_v in zip(rot_val):
        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(r_v)
      loc_dep_prob = 1
      for i in range(3):
        loc_dep_prob = min([loc_dep_prob, rot_prob[i], cent_prob[i])

      #################################
      # LOCATON INDEPENDENT PROBABILITY
      #################################
      # max & min changes over time, but max-min typically doesn't
      # 1 stdev -> 68%; 2 stdev -> 95%; 3 stdev -> 99.7%
      # pdf = probability distribution function
      c1min = cluster[c_id]['boundingbox'].min
      c2min = pc_cluster['boundingbox'].min
      c1max = cluster[c_id]['boundingbox'].max
      c2max = pc_cluster['boundingbox'].max
      
      min_mean, min_stddev = self.get_mean_stddev(c_id,'boundingbox_attr',"MIN_DIF_STATS") 
      max_mean, min_mean= self.get_mean_stddev(c_id,'boundingbox_attr',"MAX_DIF_STATS") 
      for c1Xmax, c2Xmax in zip(c1max, c2max):
        for c1Xmin, c2Xmin in zip(c1min, c2min):
          # x, y, z
          val = abs(c1Xmax - c2Xmax)
          bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(val)
          val = abs(c1Xmin - c2Xmin)
          bb_max_prob = scipy.stats.norm(max_mean, max_stddev).pdf(val)
      loc_indep_prob = 1
      for i in range(3):
        loc_indep_prob = min([loc_indep_prob, bb_min_prob[i], bb_max_prob[i])
        loc_dep_prob = min([loc_dep_prob, loc_indep_prob)
      loc_dep_prob = loc_indep_prob * loc_dep_prob
      return loc_dep_prob, loc_indep_prob


    def in_bounding_box(self, c_id, point):

    # associate grasp with the cluster
    def compare_location(self, c_id, pc_cluster):

    def compare_keypoints(self, c_id, pc_cluster):
      compare_kp(KP2):
      bf_matches, flann_matches

      for KP in enumerate(pc_cluster['kp']):
        KP.get_kp()
        KP.get_descriptor()

      for c_id, c in enumerate(self.cluster):
        cluster[c_id]['kp'].compare_kp(KP2):
        kp_stat_list = self.find_attr(c_id,'kp_attr',"KP_STATS")
        for i in len(kp_stat_list):
          sum, sum2, n = kp_stat_list[i]
          kp_mean = (x/n)
          kp_stddev = sqrt(sum2/n - sum*sum/n/n)

        kp_prob = scipy.stats.norm(kp_mean, kp_stddev).pdf(kp_v)

      for i,(m,n) in enumerate(bf_matches):
        if m.distance < 0.7*n.distance:
      for i,(m,n) in enumerate(flann_matches):
        if m.distance < 0.7*n.distance:

      rot_mean, rot_stddev = self.get_mean_stddev(c_id,'boundingbox_attr',"ROTATION_STATS")
      rot_val = pc_cluster['boundingbox'].rotation
      for r_v in zip(rot_val):
        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(r_v)
      loc_dep_prob = 1
      for i in range(3):
        loc_dep_prob = min([loc_dep_prob, rot_prob[i], cent_prob[i])




    def compare_shapes(self, c_id, pc_cluster):

    def combine_clusters(self, c_id, pc_cluster):
         
    def grasp_probabilities(self, grasp):
      # higher probability:
      #   medium fit (all grasps computed by plan_grasp?)
      #   near center / top of cluster (changes over time)
      #   grasp successfully tried
      #   grasp not unsuccessfully tried
      #   if pushing try a side grasp
      #   Note: grasps accumulate over time? Nearby grasps treated same?

      # Curiosity:
      #   not tried before
      #   not successfully picked up before
      #   just rotated and less than 6 rotations
      #   cluster never tried before ?
      #   all isolated and needs rotation
      #   not isolated and not yet pushed
      #   all grips unsuccessful, try push
      #   if moved after drop, if not pushed, try push
      #   if moved after drop, if not isolated, push towards side/isolation
      #   all isolated, all rotated, try flip
      #   all isolated, all rotated, all flipped, start interactions
      #   interact with new cluster: drop on top, push into

      # Dropping:
      #   isolate first
      #   after all isolated, try rotation
      #   after all rotated, drop on top or push
      #   stand on flat spots
      #   stack on flat spots (big to small)

    def assign_grasps(self, grasp_conf):
      # there should be no movement since analyze_pc
      # use rgbd to quickly find cluster? 
      # use cluster_centers to quickly find cluster? 
  
      for i, (g, prob) in enumerate(grasps):
        # find close cluster centers
        center = running_sum / counter
        min_dist = 1     # meter
        third_prev_cluster = None
        prev_min_cluster = None
        min_cluster  = None
        for j, c in enumerate(self.cluster_centers):
          if self.in_bounding_box(c, g):
          dist = distance_3d(c, g)
          if dist < min_dist:
            third_prev_cluster = second_cluster
            second_cluster = first_cluster
            min_dist  = dist
            first_cluster  = c
        for cluster in (first_cluster, second_cluster, third_cluster):
          for i, pc in enumerate(cluster_shape):
            if grasp
        for cluster in (first_cluster, second_cluster, third_cluster):

    def analyze_grasp(self, pc):
      self.state = "ANALYZE_GRASP"
      # find cluster that was grabbed / rotated
      if DISPLAY_PC_CLUSTERS:
        self.pc_cluster_pub = rospy.Publisher(PC_CLUSTER_TOPIC, PointCloud2, queue_size=1)

    def publish(self):
      if not DISPLAY_PC_CLUSTERS:
        return
      color = 4294967294 / 2
      clusters_pc = []
      for n in range(self.num_clusters):
        color = color / 2
        cluster_pc = [[p[0],p[1],p[2],color] for p_i, p in enumerate(self.cluster_PC[n])]
        clusters_pc.append(cluster_pc)
      cluster_pc = np.reshape(cluster_pc, (len(cluster_pc), 4))
      fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # PointField('rgba', 12, PointField.UINT32, 1)]
                PointField('rgb', 12, PointField.UINT32, 1)]
      cluster_pc = point_cloud2.create_cloud(self.pc_header, fields, cluster_pc)
      self.pc_cluster_pub.publish(cluster_pc)

    #####

    def analyze_lift(self, pc):
      self.state = "ANALYZE_LIFT"

    def analyze_drop(self, pc):
      self.state = "ANALYZE_DROP"

    # world?
    def num_clusters(self, pc):
      pass

    # world?
    def empty(self):
      pass

    def unchanged(self, cl1, cl2):
      pass

    # after drop, does it move?
    def moved(self):
      pass


    # center? on top?
    def best_grasp(self):
      pass

    def is_isolated(self):
      pass

    def analyze_pc(self, pc, KP):
      pass

    def get_center(self):
      pass

    def rolls:
      pass

    #######

    def save_history(self):
      pass

    def compare_cluster(self, CL2):
      pass

    def cluster_contains(self, grasp):
      pass

    def rotate_angle(self):
      return DEG20_IN_RADIANS

    def get_action_mode(self):
      pass

    def analyze_object(self):
      pass

    def flat:
      # center
      pass

    def compare_clusters(self, state1, state2):
      pass

    def attributes(self):
      pass

    def keypoints(self):
      pass

    def container(self):
      pass

    def cylinder(self):
      pass

    def sphere(self):
      pass

    ###################

    def combine_clusters(CL2):
      import numpy as np
      import time
      import icp

      mean_error = np.mean(distances)

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
  
    def random_select(self, pc, num_pts)
      np.random.shuffle(pc)
      if pc.shape[0] > num_pts:
          pc2 = pc[:num_pts]
      return pc2

    # icp wants same # points
    def make_same_num_points(A, B)
      if A.shape[0] > B.shape[0]:
        AA = random_select(A, B.shape[0])
        BB = B
      elif A.shape[0] < B.shape[0]:
        BB = random_select(B, A.shape[0])
        AA = A
      else
        AA = A
        BB = B
      return AA,BB
  
  
