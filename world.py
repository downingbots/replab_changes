#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
from sensor_msgs.msg import (Image)
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import DBSCAN

from utils_grasp import *
from config_grasp import *
from keypoint import *
from obb import *
from cluster import *
import utils_grasp
import cv2 as cv
from math import sqrt
import scipy.stats

class WorldState:

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
      self.world_state = 'UNINITIALIZED' 
                        # ANALYZE PLAYGROUND, ANALYZE CLUSTERS, 
                        # ISOLATE, INTERACT
      self.num_clusters = 0
      self.clusters = []

    def create_cluster(self, id = None, status = None, centroid = None, 
               shape = None, shape_attr = None, bounding_box = None,
               kp = None, kp_attr = None,
               num_locations = None, location = None, 
               num_grasps = None, grasps = None, grasp_attr = None,
               interaction = None, interaction_attr = None,
               state = None): 
      # self.cluster = {}  # done during init
      new_cluster = ClusterState()
      new_cluster.create_cluster(id, status,centroid, shape, shape_attr, bounding_box, kp, kp_attr, num_locations, location, num_grasps, grasps, grasp_attr, interaction, interaction_attr, state)
      self.clusters.append(new_cluster)

    ###############################
    # WORLD APIs : persistent state
    ###############################
    #
    # Approach:
    #   Initialize based upon first image:
    #     Filter out tray from 3d image
    #     Generate 2d image from 3d image
    #     Compute and detect KPs from 2d image
    #     Ensure KPs are in 3d image
    #     Segment filtered 3d image into clusters
    #     Using keypoints, store normalized keypoints into associated cluster
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

    def initialize_world(self, pc_clusters):
      # copy over initial analyzed pc_clusters
      self.state = 'ANALYZE PLAYGROUND'
      self.world_state = 'ANALYZE PLAYGROUND'
                        # ANALYZE PLAYGROUND, ANALYZE CLUSTERS,
                        # ISOLATE, INTERACT
      self.num_clusters = pc_clusters.num_clusters
      self.clusters = pc_clusters.clusters

#      # pc_rgb should be from executer.get_pc_rgb()
#      # get_pc_rgb filters out tray from 3d pc
#
#      # Generate 2d image from 3d image
#      rgb_2d,depth,pc_map = rgb_depth_map_from_pc(pc_rgb, fill=False)
#      # Compute and detect KPs from 2d image
#      KPs = Keypoints(rgb_2d)
#
#      # Segment filtered 3d image into clusters
#      # and ensure KPs are in 3d cluster shapes 
#      self.analyze_pc(pc, KP = KPs):
#
#      # Store intial cluster_PCs as a persistent stored world cluster
#      self.save_world()


    # store in file
    def save_world(self):
      pass

    # read from file
    def retore_world(self):
      pass

    def add_world_cluster(self, cluster):
      if self.cluster['id'] == None:
        self.cluster['id'] = self.num_clusters
        self.num_clusters += 1
      self.world.append(cluster)

    # integrates temporary "pc cluster" into persistent "world" cluster
    def integrate_current_pc_into_world(self, pc_clusters, KPs):
      if self.world_state == 'UNINITIALIZED' or len(self.clusters) == 0:
        print("copy over pc clusters")
        # just store the pc_clusters 
        self.clusters = pc_clusters
        self.save_world()
        self.world_state = 'ANALYZE PLAYGROUND' 
        return
      # compare PC to world clusters
      #    Does PC or W have more clusters?
      #    if obb match including orientation, return % match
      #    if obb cluster not there:
      #        Did it move?
      #        Is it smaller part of a new cluster in a different location?
      #        Did it become part of a bigger cluster in same location?

      # Do compare of current PC to each cluster
      kp_clust, kp_prob = self.compare_keypoints2(pc_clusters, KPs)
      # iterate through each pc_cluster and integrate into persistent state
      CLUSTER_MATCH = .75
      max_prob_clust = [0 for i, pc_c in enumerate(pc_clusters.clusters)]
      pc_cluster_match = [None for i, pc_c in enumerate(pc_clusters.clusters)]
      max_prob_cid = [None for i, pc_c in enumerate(pc_clusters.clusters)]
      w_cluster_match = [None for w_c_id, w_c in enumerate(self.clusters)]
      w_cluster_centroid_dist = [BIGNUM for w_c_id, w_c in enumerate(self.clusters)]
      for i, pc_c in enumerate(pc_clusters.clusters):
        min_centroid_dist = BIGNUM
        min_clust_dist = BIGNUM
        matching_clusters = None
        # world
        for w_c_id, w_c in enumerate(self.clusters):
          # compare bounding box, location, keypoints, shape
          centroid_dist, clust_dist = self.cluster_approx_distance(w_c_id, pc_c)
          if min_centroid_dist < centroid_dist:
            matching_clusters = [w_c_id, i]
            min_centroid_dist = min(centroid_dist, min_centroid_dist)
            min_clust_dist = min(clust_dist, min_clust_dist)
          if clust_dist < 0: 
            bb_prob, loc_indep_prob = self.compare_bounding_box(w_c_id, pc_c)
            # shape_prob = self.compare_shapes(w_c_id, pc_c)
            total_prob = 0
            if  bb_prob!=None and loc_indep_prob!=None and kp_prob!=None :
              # total_prob = bb_prob*loc_indep_prob*kp_prob
              print("cluster probs: ", w_c_id, bb_prob,loc_indep_prob,kp_prob,total_prob)
            # np....[c,pc] = [bb_prob, loc_prob, kp_prob, total_prob]
            if max_prob_clust[i] < total_prob:
              max_prob_clust[i] = total_prob
              max_prob_cid[i] = w_c_id
        if matching_clusters != None:
          print(i, " matching_clusters", matching_clusters, " centroid_dist", min_centroid_dist, " min_clust_dist", min_clust_dist)
          pc_cluster_match[i] = matching_clusters[0] # w_c_id
          pc_cluster_centroid_dist[i] = min_centroid_dist
          w_cluster_match[w_c_id] = i
        else:
          # print("pc", i, " doesn't match KPs with w", self.cluster['id'])
          print("pc", i, " doesn't match KPs in all w")
          pc_cluster_match[i] = None
      for w_c_id, w_c in enumerate(self.clusters):
        if w_cluster_match[w_c_id] == None:
          print("w", w_c_id, " no matching_kp_clusters in pc")
         
        #
        # ensure only 1-1 matches, may be multiple similar object
        # ARD: need to debug
      for i, pc_c in enumerate(pc_clusters.clusters):
        if max_prob_clust[i] > CLUSTER_MATCH and max_prob_cid[i] != None:
          combined_cluster = self.combine_clusters(self.clusters[max_prob_cid[i]],pc_c)
          print("combine cluster: ", len(combined_cluster))
          self.clusters[max_prob_cid[i]] = combined_cluster
        elif max_prob_cid[i] != None:
          combined_cluster = self.combine_clusters(self.clusters[max_prob_cid[i]],pc_c)
          print("Bad cluster match: ", len(combined_cluster))
          self.clusters[max_prob_cid[w_c_id]] = combined_cluster

      #
      # multiple clusters with shared attributes?
      #    same shape (cube), color?
      #
      # for unmatched clusters:
          # same # of clusters?  how many unaccounted for?
          # combine bounding boxes of unaccounted in different ways
          # did clusters get combined?
          # did clusters get split?
          # if location changed, did it just roll there?
          # set probabilities?
          #
          # compute a probability same
          #
          # compute probability that 2 clusters were "joined" together
          #

    # prev pc vs latest pc
    def evaluate_world_action(self, grasp):
      pass

    # world
    def record_world_action(self):
      # self.action_mode = "ISOLATE"     # ISOLATE, INTERACT, CURIOSITY
      # eval_grasp_action['EVA_SUCCESS']: True, False
      # eval_grasp_action['EVA_CLOSURE']: True, False
      # eval_grasp_action['EVA_ACTION'] = "ROTATE" "FLIP" "RANDOM_DROP"
      #     "ISOLATED_DROP" "PLAYGROUND_DROP" "PUSH" "RETRY_GRASP"
      # eval_grasp_action['EVA_DEG']
      # eval_grasp_action['EVA_REWARD']
      # eval_grasp_action['EVA_POSE']
      # eval_grasp_action['EVA_NEW_POSE']
      pass


    # world 
    def world_grasp_probabilities(self, grasp):
      pass
      # pregrasp: assign probability and recommended actions that grasp 
      #           should be chosen
      #    - returns probabilities and recommended actions
      #
      # vs assign_grasps(self, grasp_conf?
      #    - assign grasp to a cluster
      #    - calls this function, cluster.grasp_probabilities()
      # vs analyze_grasp(self, pc):
      #    - find cluster that was grabbed / rotated based upon point clouds
      # vs evaluate_grasp(grasp, e_i)
      #    - post-grasp, determine action
      #
      # eval_grasp_action = policy.evaluate_grasp(grasp, e_i)
      # self.action_mode = "ISOLATE"     # ISOLATE, INTERACT, CURIOSITY
      # eval_grasp_action['EVA_SUCCESS']: True, False
      # eval_grasp_action['EVA_CLOSURE']: True, False
      # eval_grasp_action['EVA_ACTION'] = "ROTATE" "FLIP" "RANDOM_DROP" 
      #     "ISOLATED_DROP" "PLAYGROUND_DROP" "PUSH" "RETRY_GRASP"
      # eval_grasp_action['EVA_DEG'] 
      # eval_grasp_action['EVA_REWARD']
      # eval_grasp_action['EVA_POSE'] 
      # eval_grasp_action['EVA_NEW_POSE'] 



      # ARD: TODO
      # 
      # set up "playground" area that can be easily seen by camera unobstructed
      # by arm. Move objects into playground to rotate, flip, etc.
      # Move objects from playground to isolated spots when done.

      # higher probability:
      #
      #   if multiple clusters in playground:
      #     isolate until one
      #
      #   not fully explored and in playground
      #   medium fit (all grasps computed by plan_grasp?)
      #   near center / top of cluster (changes over time)
      #   grasp successfully tried
      #   grasp not unsuccessfully tried
      #   if pushing try a side grasp
      #   Note: grasps accumulate over time? Nearby grasps treated same?

      # much of this logic belongs in evaluate_grasp(grasp, e_i):
      # evaluate grasp:
      #   did gripper close
      #   determine next action
      # evaluate_world
      #   before / after - did flip/rotate/drop/etc work?
      #   record action, cluster object
      #
      # Curiosity:
      #   in playground and not fully rotated 7 times
      #   in playground and not flipped
      #   in playground and not pushed
      #   in playground and fully investigated, move to isolation
      #   not in playground and:
      #     not tried before
      #     not successfully picked up before
      #     cluster never tried before ?
      #     not isolated and needs rotation
      #     not isolated and not yet pushed
      #     all grips unsuccessful, try push
      #
      #   if moved after drop, if not pushed, try push
      #   if moved after drop, if not isolated, push towards side/isolation
      #   all isolated, all rotated, all flipped, start interactions
      #   interact with new cluster: drop on top, push into
      #
      # dropping logic belongs in evaluate_grasp(grasp, e_i):
      # Dropping:
      #   isolate first
      #   after all isolated, try rotation
      #   after all rotated, drop on top or push
      #   stand on flat spots
      #   stack on flat spots (big to small)
      #
#      find all grasps and their clusters in playground
#        bounding box all, or partially in playground?
#        center in playground?
#        find closest cluster to playground?
#        1/2" gap between the bounding boxes?
#         cluster_approx_distance(self, c) > .5 * INCH
#      #
#      for each cluster in playground:
#        count # grasps
#          start with equal weights
#          rank by height (1 to #g)
#          rank by center by either w/l axis
#          prob = (2*h + w + l) / tot(2*h+w+l)
        #
        # s = count # successful  (rank higher)
        # f = count # failed  (rank lower)
        # mf = max # failed  (rank lower)
        # if (p == 0):
           # score[i] = mf + 1
        # else:
           # score[i] = 2*s - f + mf + 1
        #
        # prob = score * (2*h + w + l) / sum( score * (2*h + w + l))
        #
      # What if no grasp in cluster in playground?
      #   need to push
      #
      #   not fully explored and in playground
      #   medium fit (all grasps computed by plan_grasp?)
      #   near center / top of cluster (changes over time)
      #   grasp successfully tried
      #   grasp not unsuccessfully tried
      #   if pushing try a side grasp
      #   Note: grasps accumulate over time? Nearby grasps treated same?
      #
      # if grasp_in_playground():
      #   if multiple clusters in playground:
      #     isolate until one


    ###############################
    # LATEST PC APIs
    ###############################
    def deep_copy_kp(self, KP):
      # copy the list of keypoints
      # normalize the keypoints
      return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1], 
            _size = k.size, _angle = k.angle, 
            _response = k.response, _octave = k.octave, 
            _class_id = k.class_id) for k in f]

    def publish_pc(self):
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


    #####################
    # BOTH UTILITIES (world & latest pc)
    #####################
    def find_epsilon(self, pc):
      from rotor import Rotor
      from sklearn.neighbors import NearestNeighbors

      neigh = NearestNeighbors(n_neighbors=2)
      nbrs = neigh.fit(pc)
      distances, indices = nbrs.kneighbors(pc)
      distances = np.sort(distances, axis=0)
      print("distances len:", len(distances))
      for i in range(len(distances[:])):
        distances[i][0] = i
        # if distances[i][0] > 0 or distances[i][1] > 0:
          # print("distances: ",i,distances[i])
      # distances = distances[:,1]
      rotor = Rotor()
      rotor.fit_rotate(distances)
      elbow_index = rotor.get_elbow_index()
      # print("distances: ",distances)
      # distance1 = distances[:,1]
      # distance2 = distances[1,:]
      # print("elbow: ",elbow_index, distance1[elbow_index], distance2[elbow_index])
      # print("epsilon: ",elbow_index, distances[elbow_index])
      return distances[elbow_index][1]

    # analyze latest point cloud into a set of clusters
    def analyze_pc(self, pc, KP = None, min_sample_sz = CLUSTER_MIN_SZ):
      # db1 = DBSCAN(eps=.001, min_samples=3,
      # db1 = DBSCAN(eps=.01, min_samples=3,
      clust_eps = self.find_epsilon(pc)
      print("clust eps: ",clust_eps)
      # clust_eps_delta = CLUSTER_EPS_DELTA
      # min_sample_size = min_sample_sz
      min_sample_sz = CLUSTER_MIN_SZ
      # for i in range(CLUSTER_MIN_SZ - 8):
      for i in range(1):
        # clust_eps = CLUSTER_EPS
        min_sample_size = min_sample_sz - i
        # for j in range(10):
        for j in range(1):
          db1 = DBSCAN(eps=clust_eps, min_samples=min_sample_size,
                       n_jobs=-1).fit(pc)
          # Number of clusters in labels, ignoring noise if present.
          n1_clusters = len(set(db1.labels_)) - (1 if -1 in db1.labels_ else 0)
          print("DBSCAN: # ", n1_clusters, min_sample_size, clust_eps)
          if n1_clusters > 0:
            break
          # else:
            # clust_eps += CLUSTER_EPS_DELTA
        if n1_clusters > 0:
          break
      if n1_clusters == 0:
        return False
      # self.clusters should be empty list. Assert?
      # cluster = []
      for i in range(n1_clusters):
        # Add a new cluster for the pc. 
        # Later, combine clusters with existing cluster?
        self.create_cluster(id = i,kp = [], shape = [])
      if KP != None:
        kp = KP.get_kp() 

    # for cluster in set(labels):
    #   if cluster != -1:
    #     for i, label in enumerate(labels):
    #       if labels[i] == cluster:
      for c in set(db1.labels_):
        if c != -1:
          running_sum = np.array([0.0, 0.0, 0.0, 0.0])
          counter = 0
          # for i,p in enumerate(pc):
          for i, label in enumerate(db1.labels_):
              if db1.labels_[i] == c:
                  running_sum += pc[counter]
                  counter += 1
                  clust = self.clusters[c].cluster
                  # print("clust ",c," append pc",i," len", len(self.clusters[c].cluster['shape']), len(clust['shape']))
                  # print("clust ",c," append pc",i," len", len(clust['shape']), len(clust['shape']))
                  clust['shape'].append(pc[counter])
                  if KP != None:
                    # map keypoints to cluster before normalizing shape
                    if [pc[counter][0], pc[counter][1]] in KP.get_kp():
                      self.clusters[c].cluster['kp'].append([pc[counter],KP.get_descriptor()])
                      print("kp found in cluster",c)
          center = running_sum / counter
          # ClusterState: [id, center, KPs, pc]
          # ARD: TODO : do we want 'center' to be pre-normalized 'centroid'
          self.clusters[c].cluster['center'] = center
          # print("center for clust", c , " is ", self.clusters[c].cluster['center'])
      for i, c in enumerate(self.clusters):
        # ARD: store normalized cluster and unnormalized cluster?
        # normalize shape
        # print("Cluster ",i," len", len(c.cluster['shape']))
        c.normalize()
        # need to normalize KP:
        # cluster[c].cluster['kp'].append(pc[i])
        # compute bounding box
        print("cluster ",i," len", len(c.cluster['shape']))
        c.compute_bounding_box()
        # print("cluster", i, " obb min,max,rot is ", c.cluster['obb'].min, c.cluster['obb'].max, c.cluster['obb'].rotation)
        # print("cluster", i, " centroid is ", c.cluster['obb'].centroid)
        # print("cluster", i, " points are ", c.cluster['obb'].points)
      # return cluster
      return True

    #####################
    # BOTH UTILITIES
    #####################
    def in_playground(self, point):
      pass
      # grasp in playground?
      # cluster center in playground?

    # ARD TODO: use self.clusters[c_id]
    # return bounding_box_in_playground, centroid_in_playground
    def bounding_box_in_playground(self):
      # for c in self.clusters:
        bb = self.cluster['bounding_box'] 
        # bb = c.cluster['bounding_box'] 
        min_x,min_y = PLAYGROUND[0]
        max_x,max_y = PLAYGROUND[1]
        for i in range(3):
          x,y,z = bb[i]
          if not (x >= min_x and x <= max_x
            and y >= min_y and y <= max_y):
              return False
              # continue
        return True
      # return False

    # ARD TODO: use self.clusters[c_id]
    def centroid_in_playground(self):
      centroid = self.cluster['centroid']
      min_x,min_y = PLAYGROUND[0]
      max_x,max_y = PLAYGROUND[1]
      if (centroid.x >= min_x and centroid.x <= max_x
         and centroid.y >= min_y and centroid.y <= max_y):
            return True
      else:
            return False

    # ARD TODO: use self.clusters[c_id]
    def cluster_approx_distance(self, c_id, c):
      # simple heuristic to determine minimum distance between objects.
      # the computed distance assumes worst-case rotation of clusters.
      w_obb = self.clusters[c_id].cluster['obb'] 
      pc_obb = c.cluster['obb'] 
      w_centroid = w_obb.centroid
      pc_centroid = pc_obb.centroid
      cent_dist = distance_3d(w_centroid, pc_centroid)
      w_bb_dist1 = distance_3d(w_centroid, w_obb.min)
      pc_bb_dist2 = distance_3d(pc_centroid, pc_obb.min)
      # neg dist_between_clust means likely the same cluster
      dist_between_clust = cent_dist - w_bb_dist1 - pc_bb_dist2
      return cent_dist, dist_between_clust 

    def find_world_cluster(self, id = None, center = None, shape = None, 
                           kp = None, location = None, state = None):
      pass

    # def grasp_in_playground(self):
      # self.cluster['centroid'] = centroid          # current centroid 
      # self.cluster['bounding_box'] = bounding_box  # 3 sets of (x,y,z)
      # pass

    #########################################
    # UTILITIES FOR INTEGRATING LATEST PC INTO WORLD 
    #########################################

    def get_pct_overlap(self, obb1, obb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Axis-aligned bounding box only.
    
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    
        Returns
        -------
        float
            in [0, 1]
        """
        bb1 = {}
        bb2 = {}
        [bb1['x1'], bb1['y1'], bb1['z1'] ] = obb1.min
        [bb1['x2'], bb1['y2'], bb1['z2'] ] = obb1.max

        [bb2['x1'], bb2['y1'], bb2['z1'] ] = obb2.min
        [bb2['x2'], bb2['y2'], bb2['z2'] ] = obb2.max
        # Axis-aligned bounding box only.
        # If we normalize the OBBs, will the bb be axis aligned?
        # if abs(obb1.rotation - obb2.rotation) / ((obb1.rotation + obb2.rotation)/2) < .05:
        #   print("rotations don't match", obb1.rotation, obb2.rotation)
        # if obb1.rotation != obb2.rotation:
        # print("rotations1:", obb1.rotation)
        # print("rotations2:", obb2.rotation)

        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb1['z1'] < bb1['z2']

        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
        assert bb2['z1'] < bb2['z2']
    
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        z_up = max(bb1['z1'], bb2['z1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
        z_down = min(bb1['z2'], bb2['z2'])
    
        if x_right < x_left or y_bottom < y_top or z_down < z_up:
            return 0.0
    
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top) * (z_down - z_up)
    
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']) * (bb1['z2'] - bb1['z1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']) * (bb2['z2'] - bb1['z1'])
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

        """
    from shapely.geometry import Polygon

    def pcnt_overlap(self, obb1, obb2):
       poly_1 = Polygon(obb1.points)
       poly_2 = Polygon(obb2.points)
       iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
       return iou

        Return true if a box overlaps with this box.
        A Box class has 8:
        self.left                           left side
        self.top                            "top" side
        self.right                          right side
        self.bottom                         bottom side
        self.lt                             left-top point
        self.rb                             right-bottom point
        self.rt =                           right-top point
        self.lb                             left-bottom point

        A point is an x,y coordinate. 
        self.points[0] > 

        other.points 

        if(self.lt.x > other.rb.x or other.lt.x > self.rb.x):
            return False
        elif(self.rb.y < other.lt.y or other.rb.y < self.lt.y):
            return False
        else:
            return True
        """


    def compare_bounding_box(self, w_c_id, pc_cluster):
      # LOCATON DEPENDENT PROBABILITY
      # centroid changes when moved
      # yet, stddev still can be computed?
      # get rotation based on bounding box
      # ARD: TODO
      cent_mean, cent_stddev = self.clusters[w_c_id].get_mean_stddev('obb_attr',"CENTROID_STATS") 
      # print("pc_cluster obb:", pc_cluster.cluster['obb']) # not informative
      if pc_cluster.cluster['obb'] == None or self.clusters[w_c_id].cluster['obb'] == None:
        if pc_cluster.cluster['obb'] == None or self.clusters[w_c_id].cluster['obb'] == None:
          print("no OBB for pc", pc_cluster.cluster['id'])
        if self.clusters[w_c_id].cluster['obb'] == None:
          print("no OBB for w", w_c_id)
        return 0,0
      # percent overlap
      pct_ovrlp = self.get_pct_overlap( self.clusters[w_c_id].cluster['obb'],pc_cluster.cluster['obb'])
      if pct_ovrlp > 0:
        print("obb overlap: world",w_c_id,pct_ovrlp)
      #    if obb match including orientation, return % match
      #    if obb cluster not there:
      #        Did it move?
      #        Is it smaller part of a new cluster in a different location?
      #        Did it become part of a bigger cluster in same location?
      #

      pc_cent_val = pc_cluster.cluster['obb'].centroid
      w_cent_val = self.clusters[w_c_id].cluster['obb'].centroid
      # if unmoved, computed mean and variance centroid 
      for c_v in zip(w_cent_val, pc_cent_val):
        if cent_mean != None and cent_stddev != None:
          cent_prob = scipy.stats.norm(cent_mean, cent_stddev).pdf(c_v)
        else:
          cent_prob = 0
      # rotation changes when moved
      # if unmoved, computed mean and stddev centroid 
      rot_mean, rot_stddev = self.clusters[w_c_id].get_mean_stddev('obb_attr',"ROTATION_STATS") 
      rot_val = pc_cluster.cluster['obb'].rotation
      # for r_v in zip(rot_val):
      if rot_mean != None and rot_stddev != None:
        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(rot_val)
      else:
        rot_prob = 0
      # loc_dep_prob = 1
      # for i in range(3):
      #   loc_dep_prob = min(loc_dep_prob, rot_prob[i], cent_prob[i])
      loc_dep_prob = max(pct_ovrlp, rot_prob, cent_prob)
      #
      # LOCATON INDEPENDENT PROBABILITY
      # max & min changes over time, but max-min typically doesn't
      # 1 stdev -> 68%; 2 stdev -> 95%; 3 stdev -> 99.7%
      # pdf = probability distribution function
      c1min = self.clusters[w_c_id].cluster['obb'].min
      c2min = pc_cluster.cluster['obb'].min
      c1max = self.clusters[w_c_id].cluster['obb'].max
      c2max = pc_cluster.cluster['obb'].max
      #
      min_mean, min_stddev = self.clusters[w_c_id].get_mean_stddev('obb_attr',"MIN_DIF_STATS") 
      max_mean, min_mean= self.clusters[w_c_id].get_mean_stddev('obb_attr',"MAX_DIF_STATS") 
      for c1Xmax, c2Xmax in zip(c1max, c2max):
        for c1Xmin, c2Xmin in zip(c1min, c2min):
          # x, y, z
          val = abs(c1Xmax - c2Xmax)
          if min_mean != None and min_stddev != None:
            bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(val)
          else:
            bb_min_prob = 0
          val = abs(c1Xmin - c2Xmin)
          if max_mean != None and max_stddev != None:
            bb_max_prob = scipy.stats.norm(max_mean, max_stddev).pdf(val)
            print("bb_max_prob:", bb_max_prob, bb_min_prob)
          else:
            bb_max_prob = 0
          # print("bb_max_prob:", bb_max_prob, bb_min_prob)
      loc_indep_prob = 1
      # for i in range(3):
      #   loc_indep_prob = min(loc_indep_prob, bb_min_prob[i], bb_max_prob[i])
      loc_indep_prob = min(bb_min_prob, bb_max_prob)
      loc_dep_prob = loc_indep_prob * loc_dep_prob
      return loc_dep_prob, loc_indep_prob

    # associate grasp with the cluster
    def compare_location(self, c_id, pc_cluster):
      pass

    # for integration into world cluster
    # def compare_keypoints(self, c_id, pc_cluster):
    def compare_keypoints(self, pc_cluster, KPs):
      KPList = pc_cluster.cluster['kp'].get_kp()
      print("KPList",KPList)
      for kp_i, KP in enumerate(KPList):
        # KP.get_kp()
        # KP.get_descriptor()
        # return list_cluster[0],score2

        cluster_match, score = self.clusters[c_id].cluster['kp'].compare_kp(pc_cluster)
        # just use ratios
        # kp_mean, kp_stddev = self.get_mean_stddev(c_id,'kp_attr',"KP_STATS") 
        # bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(kp_distance)
        #
        # ratio test as per Lowe's paper
        # for i,(m,n) in enumerate(flann_matches):
          # ratio[i] = m.distance / n.distance
        #
        # for i,(m,n) in enumerate(bf_matches):
          # ratio[i] = m.distance / n.distance
        #
      # for i,(m,n) in enumerate(bf_matches):
        # if m.distance < 0.7*n.distance:
      # for i,(m,n) in enumerate(flann_matches):
        # if m.distance < 0.7*n.distance:
      # set_attr(self, c_id, field, key_value):
      #
      rot_mean, rot_stddev = self.clusters[c_id].get_mean_stddev('obb_attr',"ROTATION_STATS")
      rot_val = pc_cluster['obb'].rotation
      for r_v in zip(rot_val):
        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(r_v)
      loc_dep_prob = 1
      for i in range(3):
        loc_dep_prob = min(loc_dep_prob, rot_prob[i], cent_prob[i])
      # ARD: rotation???
      return kp_prob 

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
    # World
    #################

    def assign_grasps(self, grasp_conf):
      # there should be no movement since analyze_pc
  
      found = False
      if grasp_conf == None:
        return False
      for i, (g, prob) in enumerate(grasp_conf):
        # if no bounding box, find close cluster centers
        # center = running_sum / counter
        # min_dist = 1     # meter
        # third_prev_cluster = None
        # prev_min_cluster = None
        # min_cluster  = None
        # print("clusters:",self.clusters)
        # print("clusters[0]:",self.clusters[0])
        # print("clusters[0] cenroid:",self.clusters[0].cluster['centroid'])
        # for c in self.clusters:
        d1 = [BIGNUM, BIGNUM]
        d2 = [BIGNUM, BIGNUM]
        d3 = [BIGNUM, BIGNUM]
        # print("len clusters:", len(self.clusters))
        for c_id, c in enumerate(self.clusters):
         # for j, cc in enumerate(cluster_centers):
         # g has [x,y,z,theta]
         # ret = ClusterState.in_bounding_box(c, [g[0],g[1],g[2]])
         ret = c.in_bounding_box([g[0],g[1],g[2]])
         if ret == None or ret == False:
            # cluster_center = c.cluster['centroid']
            cluster_center = c.cluster['obb'].centroid
            g_cent_dist = distance_3d(cluster_center, g)
            # print(" c_id ", c_id," centroid is ", cluster_center," vs dist ", g_cent_dist)
            if g_cent_dist < d1[1]:
              d3 = d2
              d2 = d1
              d1 = [c_id, g_cent_dist]
              print("No OBB; closest to center: ", d1, d2, d3)
            elif g_cent_dist < d2:
              d3 = d2
              d2 = [c_id, g_cent_dist]
            elif g_cent_dist < d2:
              d3 = [c_id, g_cent_dist]
            continue
         else:
            print("grip ",g, " in obb for ",c_id)
            # ARD TODO: transform to take flip/rotate into consideration
            c.cluster['num_grasps'] += 1
            c.cluster['grasps'].append(g)
            c.cluster['grasp_attr'] = []            # ["name",value] pair
            found = True
         continue
        if not found:
          print("no cluster found for grasp ", g)
          print("best fits ", d1,d2,d3)
        continue
      # Return doesn't matter
      return found
# END:for i, (g, prob) in enumerate(grasp_conf):
        # use rgbd to quickly find cluster? 
        # use cluster_centers to quickly find cluster? 
          # if no bounding box? 
#          dist = distance_3d(c, g)
#          if dist < min_dist:
#            third_prev_cluster = second_cluster
#            second_cluster = first_cluster
#            min_dist  = dist
#            first_cluster  = c
        # for cluster in (first_cluster, second_cluster, third_cluster):
          # for i, pc in enumerate(cluster_shape):
            # if grasp
        # for cluster in (first_cluster, second_cluster, third_cluster):

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

    def compare_keypoints2(self, pc_c, KP):
      kp_prob = []
      # print("KP:",KP.get_kp())
      for i in range(self.num_clusters):
        kp_prob.append(None)
     
      matching_pc_cluster = []
      score = []
      for w_c_id, c in enumerate(self.clusters):
        matching_pc_cluster.append(None)
        score.append(None)
      for i, c in enumerate(self.clusters):
        # get pointcloud associated with designated cluster
        cluster_pc = c.cluster['shape']
        # Generate 2d image from 3d pointcloud
        rgb_pc = cluster_pc
        # print(cluster_pc[0])
        cluster_img,depth,pc_map, pc_img = rgb_depth_map_from_pc(cluster_pc, rgb_pc, fill=False)
        # return rgb, depth, pc_map, pc_img
        # cluster_img is x=RGB_WIDTH/y=RGB_HEIGHT/val=RGB

        # Compute and detect KPs from 2d image of single world cluster
        cluster_kp = Keypoints(cluster_img)
        print("num cluster KPs: ", len(cluster_kp.get_kp()))
        c.cluster['kp'] = cluster_kp

        # compare cluster's KPs to full set of KPs from latest image 
        matching_pc_cluster[i], score[i] = cluster_kp.compare_kp(pc_c, KP)
        if matching_pc_cluster[i] != None or score[i] > 0:
          print("w",i, "matching keypoint pc",matching_pc_cluster[i]," score ", score[i])
      return matching_pc_cluster, score


#     Map new KPs to new clusters
#     Map new clusters to world clusters
#     compare and combine clusters
#     Future: handle rotation, flipping, rolling

