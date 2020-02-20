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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

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
      self.clusters_history = []
      self.kp_info = []
      self.KP = None  
      # self.obb_pub = rospy.Publisher('/obb', Marker, queue_size=1)
      self.obb_pub = rospy.Publisher('/obb', MarkerArray, queue_size=1)

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

    def replace_world_clusters(self, pc_clusters):
        print("copy over pc clusters")
        # just store the pc_clusters
        self.clusters = pc_clusters.clusters
        self.save_world()
        self.world_state = 'ANALYZE PLAYGROUND'
        return

    # integrates temporary "pc cluster" into persistent "world" cluster
    def integrate_current_pc_into_world(self, pc_clusters, pc_kp_info):
      if self.world_state == 'UNINITIALIZED' or len(self.clusters) == 0:
        self.compare_all_keypoints(pc_clusters)
        print("copy over pc clusters")
        # just store the pc_clusters 
        self.clusters_history.append(self.clusters)
        self.clusters = pc_clusters
        self.kp_info_history.append(self.pc_kp_info)
        self.kp_info = pc_kp_info
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


      # iterate through each pc_cluster and integrate into persistent state
      centroid_matching_clusters = [None for w_c_id, w_c in enumerate(self.clusters)]
      centroid_min_dist = [BIGNUM for w_c_id, w_c in enumerate(self.clusters)]
      centroid_min_clust_dist = [BIGNUM for w_c_id, w_c in enumerate(self.clusters)]
      obb_matching_clusters = [None for i, w_c in enumerate(self.clusters)]
      obb_max_pct_ovlp = [0 for i, w_c in enumerate(self.clusters)]
      obb_ovlp_cnt = [0 for i, w_c in enumerate(self.clusters)]

      ##############
      # Compare KPs
      # Returns an array indexed by w_c_id to find matching pc_c_ids
      KPs = pc_clusters.KP
      kp_matching_pc_clusters, kp_distance, kp_pts = self.compare_keypoints(KPs)
      self.kp_info = [kp_matching_pc_clusters, kp_distance, kp_pts]

      ##############

      # Do compare of current PC to each cluster
      for i, pc_c in enumerate(pc_clusters.clusters):
        # world
        for w_c_id, w_c in enumerate(self.clusters):
          ##############
          # Compare Centroid/Cluster distance
          ##############
          centroid_dist, clust_dist = self.cluster_approx_distance(w_c_id, pc_c)
          # print("centroid/clust dist:", centroid_dist, clust_dist )
          if centroid_min_dist[w_c_id] > centroid_dist:
            centroid_matching_clusters[w_c_id] = i
            centroid_min_dist[w_c_id] = min(centroid_dist, centroid_min_dist)
            centroid_min_clust_dist[w_c_id] = min(clust_dist, centroid_min_clust_dist)

          ##############
          # Compare OBB overlap
          ##############
          if clust_dist <= 0: 
            # bb_prob, loc_indep_prob = self.compare_bounding_box(w_c_id, pc_c)
            obb_pct_ovlp = self.compare_bounding_box(w_c_id, pc_c)

            if obb_max_pct_ovlp[w_c_id] < obb_pct_ovlp:
              obb_matching_clusters[w_c_id] = i
              obb_max_pct_ovlp[w_c_id] = obb_pct_ovlp

            # indicator that we should combine clusters
            if obb_pct_ovlp > .70:
              obb_ovlp_cnt[w_c_id] += 1


          ##############
          # shape_prob = self.compare_shapes(w_c_id, pc_c)
          # could compare sizes, but OBB roughly does this

      #####################################
      # Integrate PC and World clusters
      #####################################
      print("#######################")
      print("# CLUSTER MATCH SUMMARY")
      print("#######################")
      print("# w/pc clusters:", len(self.clusters), len(pc_clusters.clusters))
      pc_unmatched = [i for i, pc_c in enumerate(pc_clusters.clusters)]
      w_unmatched = []
  
      # ARD PROBLEM: same pc_cluster being mapped to multiple w_clusters
      for w_c_id, w_c in enumerate(self.clusters):
        cent_match      = centroid_matching_clusters[w_c_id]
        cent_dist       = centroid_min_dist[w_c_id]
        cent_clust_dist = centroid_min_clust_dist[w_c_id]
        obb_match       = obb_matching_clusters[w_c_id]
        obb_ovrlp       = obb_max_pct_ovlp[w_c_id]
        obb_ovrlp_num   = obb_ovlp_cnt[w_c_id]
        kp_match        = kp_matching_pc_clusters[w_c_id]
        print(w_c_id,"c/obb/kp matches:", cent_match, obb_match, kp_match)
        print("c  : ", cent_match, cent_dist, cent_clust_dist)
        print("obb: ", obb_match, obb_ovrlp, obb_ovrlp_num)
        print("kp : ", kp_match)
        if cent_match in pc_unmatched:
          pc_unmatched.remove(cent_match)
        if obb_match in pc_unmatched:
          pc_unmatched.remove(obb_match)
        if kp_match in pc_unmatched:
          pc_unmatched.remove(kp_match)
      
        if cent_match == kp_match:
          if cent_clust_dist > 0:
            print("w", w_c_id, " matched and moved slightly")
          elif cent_clust_dist < 0:
            if cent_match == obb_match:
              if obb_ovrlp > .70:
                print("w", w_c_id, " matched and unmoved and overlap. Combine clusters.")
                # ARD: TODO
                # combined_cluster = self.combine_clusters(self.clusters[w_cid], pc_clusters[cent_match])
              else:
                print("w", w_c_id, " matched and possibly rotated. Transform to combine clusters.")
        else:
          if kp_match == None:
            if cent_clust_dist < 0 and cent_match == obb_match and obb_ovrlp > .95:
              print("w", w_c_id, " matched and unmoved and overlap by position and obb. Combine clusters.")
              # ARD: TODO
              # combined_cluster = self.combine_clusters(self.clusters[w_c_id], pc_clusters[cent_match])
            else:
               w_unmatched.append(w_c_id)
               print("w", w_c_id, " not matched ")
          else:
            kps = self.clusters[w_c_id].cluster['kp'].get_kp()
            print("w", w_c_id, " matched kp and moved ")
            # print("w", w_c_id, " matched kp but moved to ", kp_xy[w_d_id], " from ", kps)

      if len(pc_unmatched) > 0:
        print("unmatched PC clusters: ", pc_unmatched)
        print("unmatched PC cluster sizes: "),
        for pc_id in pc_unmatched:
          shape = pc_clusters.clusters[pc_id].cluster['shape']
          print(len(shape)),
        print("")

      min_pc_height = BIGNUM
      max_pc_c_height = []
      for pc_c in pc_clusters.clusters:
        shape = pc_c.cluster['shape']
        max_pc_height = 0
        for pt in shape:
          min_pc_height = min(min_pc_height, pt[2])
          max_pc_height = max(max_pc_height, pt[2])
        max_pc_c_height.append(max_pc_height)

      min_w_height = BIGNUM
      max_w_c_height = []
      for w_c in self.clusters:
        shape = w_c.cluster['shape']
        max_w_height = 0
        for pt in shape:
          min_w_height = min(min_w_height, pt[2])
          max_w_height = max(max_w_height, pt[2])
        max_w_c_height.append(max_w_height)
      print("min cluster heights: ", min_pc_height, min_w_height)


      LOW_VOL_THRESH = (.1 * INCH)**3
      LOW_HEIGHT_THRESH = (.1 * INCH)
      if len(pc_unmatched) > 0:
        low_pc_vol = []
        print("unmatched PC clusters: ", pc_unmatched)
        print("unmatched PC cluster sizes: "),
        for pc_id in pc_unmatched:
          shape = pc_clusters.clusters[pc_id].cluster['shape']
          obb = pc_clusters.clusters[pc_id].cluster['obb']
          vol = None
          c_height = (max_pc_c_height[pc_id] - min_pc_height)
          if obb != None:
            vol = OBB.obb_volume(obb)
            if vol == None or (vol < LOW_VOL_THRESH and c_height < LOW_HEIGHT_THRESH):
              low_pc_vol.append(pc_id)
          print("(",len(shape), vol, c_height, ")"),
        print(" ")
        print("Low volume pc_id that maybe should be ignored: ", low_pc_vol)
        # ARD TODO:
        # print("add to W clusters")
        # was there a split of a w cluster?
        # add to w clusters?
        # were w clusters combined into bigger cluster")
      if len(w_unmatched) > 0:
        print("unmatched W clusters : ", w_unmatched)
        print("unmatched W cluster sizes: "),
        low_w_vol = []
        for w_id in w_unmatched:
          shape = self.clusters[w_id].cluster['shape']
          obb = self.clusters[w_id].cluster['obb']
          vol = None
          c_height = (max_w_c_height[w_id] - min_w_height)
          if obb != None:
            vol = OBB.obb_volume(obb)
            # only seeing top?
            if vol == None or (vol < LOW_VOL_THRESH and c_height < LOW_HEIGHT_THRESH):
              low_w_vol.append(w_id)
            # vol = self.clusters[w_id].cluster['obb'].obb_volume
          print("(",len(shape), vol, c_height, ")"),
        print(" ")
        print("Low volume w_id that maybe should be ignored: ", low_w_vol)
      # ARD: for debugging, compare consecutive 2 clusters as moving window
      self.replace_world_clusters(pc_clusters)
        # ARD TODO:
        # was there a split of a w cluster?
        # were there 2+ unmatched pc clusters? 
        # was this cluster previously unmatched? keep history.
       

#        #
#        # ensure only 1-1 matches, may be multiple similar object
#        # ARD: need to debug
#      for i, pc_c in enumerate(pc_clusters.clusters):
#        if max_prob_clust[i] > CLUSTER_MATCH and max_prob_cid[i] != None:
#          combined_cluster = self.combine_clusters(self.clusters[max_prob_cid[i]],pc_c)
#          print("combine cluster: ", len(combined_cluster))
#          self.clusters[max_prob_cid[i]] = combined_cluster
#        elif max_prob_cid[i] != None:
#          combined_cluster = self.combine_clusters(self.clusters[max_prob_cid[i]],pc_c)
#          print("Bad cluster match: ", len(combined_cluster))
#          self.clusters[max_prob_cid[w_c_id]] = combined_cluster

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
    def deep_copy_kp(self, KP, kp_list):
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
      j = None
      k = None
      for i in range(len(distances[:])):
        distances[i][0] = i
        if distances[i][1] == 0:
          j = i
        elif distances[i][1] > 1:
          if k == None:
            k = i
            print("eps dist below/abv 1: ", i, len(distances - i))
          # distances = distances[:(i-1)]
          # break
        # if distances[i][0] > 0 or distances[i][1] > 0:
          # print("distances: ",i,distances[i])
      if j != None:
        distances = distances[(j+1):]
      rotor = Rotor()
      rotor.fit_rotate(distances)
      elbow_index = rotor.get_elbow_index()
      # rotor.plot_elbow()
      # rotor.plot_knee()
      # print("distances: ",distances)
      # distance1 = distances[:,1]
      # distance2 = distances[1,:]
      # print("elbow: ",elbow_index, distance1[elbow_index], distance2[elbow_index])
      # print("epsilon: ",elbow_index, distances[elbow_index][1])
      return distances[elbow_index][1]

    # analyze latest point cloud into a set of clusters
    def analyze_pc(self, pc, KP = None, min_sample_sz = CLUSTER_MIN_SZ):
      self.KP = KP
      # db1 = DBSCAN(eps=.001, min_samples=3,
      # db1 = DBSCAN(eps=.01, min_samples=3,
      pc_no_rgb = np.array(pc)[:, :3]
      print("pc_no_rgb:",len(pc_no_rgb[0]))
      clust_eps = self.find_epsilon(pc_no_rgb)
      print("clust eps: ",clust_eps)
      # clust_eps_delta = CLUSTER_EPS_DELTA
      # min_sample_size = min_sample_sz
      min_sample_sz = CLUSTER_MIN_SZ
      # for i in range(CLUSTER_MIN_SZ - 8):
      for i in range(5):
        # clust_eps = CLUSTER_EPS
        min_sample_size = min_sample_sz - i
        # for j in range(10):
        for j in range(1):
          db1 = DBSCAN(eps=clust_eps, min_samples=min_sample_size,
                       n_jobs=-1).fit(pc_no_rgb)
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
        print("NO CLUSTERS FOUND")
        return False
      # self.clusters should be empty list. Assert?
      # cluster = []
      counter = []
      running_sum = []
      kp_list = []
      for n1 in range(n1_clusters):
        # Add a new cluster for the pc. 
        # Later, combine clusters with existing cluster?
        self.create_cluster(id = n1,kp = [], shape = [])
        counter.append(0)
        running_sum.append(np.array([0.0, 0.0, 0.0, 0.0]))
        # running_sum.append(np.array([0.0, 0.0, 0.0]))
        kp_list.append([])
      if KP != None:
        kp = KP.get_kp() 
        # kp_3d = KP.kp_to_3d_point(pc[i])
        # kp_3d maps kp list to a list of [3d_keypoint] from pc
        kp_3d = KP.kp_to_3d_point(pc)    # cached in KP object

      print("len labels vs len pc[i]:", len(db1.labels_), len(pc))
      for c_id in set(db1.labels_):
        if c_id != -1:
          for i, label in enumerate(db1.labels_):
              if db1.labels_[i] == c_id:
                  # print("label", c_id, i, pc[i])
                  counter[c_id] += 1
                  running_sum[c_id] += pc[i]
                  # print(c_id, "shape append", pc[i])
                  self.clusters[c_id].cluster['shape'].append(pc[i])
      for c_id, c in enumerate(self.clusters):
        # compute bounding box
        c.compute_bounding_box()
        if KP != None:
          # map 3d keypoints to cluster 
          for k, kp_3d_pnt in enumerate(kp_3d):
            # print("kp_3d_pnt:", kp_3d_pnt)
            # if int(pc[i][0]*1000000) == int(kp_3d_pnt[0]*1000000):
            #   print("pc kp: ", pc[i], kp_3d_pnt)
            obb = None
            ret = self.clusters[c_id].in_bounding_box(kp_3d_pnt)
            if ret == True:
              obb = self.clusters[c_id].cluster['obb']
              kp_list[c_id].append([k, c_id, pc[i]])
              # ('setting an array element with a sequence.',)
              print("kp found in cluster obb", k, c_id)
              shape = self.clusters[c_id].cluster['shape']
              for pt in shape:
                if pt[0] == kp_3d_pnt[0] and pt[1] == kp_3d_pnt[1] and pt[2] == kp_3d_pnt[2]:
                  kp_list[c_id].append([k, c_id, pt, obb])
                  # ('setting an array element with a sequence.',)
                  print("kp found in cluster", k, c_id, pt[0],pt[1],pt[2])
                else:
                  kp_list[c_id].append([k, None, pt, obb])
	c.cluster['kp_c_pc_mapping'] = kp_list[c_id]   # kp_3dpt, pc_c_id, pc[i]
        center = running_sum[c_id] / counter[c_id]
        c.cluster['center'] = center
        # print("center for clust", c , " is ", self.clusters[c].cluster['center'])
        # normalize shape
        c.normalize()
        # need to normalize KP:
        print("cluster ",c_id," len", len(c.cluster['shape']))
        # print("cluster", c_id, " obb min,max,rot is ", c.cluster['obb'].min, c.cluster['obb'].max, c.cluster['obb'].rotation)
        # print("cluster", c_id, " centroid is ", c.cluster['obb'].centroid)
        # print(c_id, " cluster shape:", self.clusters[i].cluster['shape'])
      print("num_pc_clusters:",len(self.clusters))
      self.publish_obb()
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
      # if dist_between_clust < 0:
      #   dist_between_clust = 0
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
        pct_overlap = OBB.obb_overlap(obb1, obb2)
        # print("pct ovrlap:", pct_overlap)
        return pct_overlap

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
      if pct_ovrlp > 0.90:
        print("obb overlap: world",w_c_id, pc_cluster.cluster['id'], pct_ovrlp)
      return pct_ovrlp

#      #    if obb match including orientation, return % match
#      #    if obb cluster not there:
#      #        Did it move?
#      #        Is it smaller part of a new cluster in a different location?
#      #        Did it become part of a bigger cluster in same location?
#      #
#
#      pc_cent_val = pc_cluster.cluster['obb'].centroid
#      w_cent_val = self.clusters[w_c_id].cluster['obb'].centroid
#      cent_dist = distance_3d(pc_cent_val, w_cent_val)
#      print("cluster centroid distance:", cent_dist)
# 
#      # if unmoved, computed mean and variance centroid 
#      for c_v in zip(w_cent_val, pc_cent_val):
#        if cent_mean != None and cent_stddev != None:
#          cent_prob = scipy.stats.norm(cent_mean, cent_stddev).pdf(c_v)
#        else:
#          cent_prob = 0
#      # ARD: rotation is matrix, and below rotation code is wrong
#      # rotation changes when moved
#      # if unmoved, computed mean and stddev centroid 
#      # rot_mean, rot_stddev = self.clusters[w_c_id].get_mean_stddev('obb_attr',"ROTATION_STATS") 
#      # rot_val = pc_cluster.cluster['obb'].rotation
#      # for r_v in zip(rot_val):
#      # if rot_mean != None and rot_stddev != None:
#        # rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(rot_val)
#      # else:
#        # rot_prob = 0
#      # loc_dep_prob = 1
#      # for i in range(3):
#      #   loc_dep_prob = min(loc_dep_prob, rot_prob[i], cent_prob[i])
#      rot_prob = 0
#      loc_dep_prob = max(pct_ovrlp, rot_prob, cent_prob)
#      #
#      # LOCATON INDEPENDENT PROBABILITY
#      # max & min changes over time, but max-min typically doesn't
#      # 1 stdev -> 68%; 2 stdev -> 95%; 3 stdev -> 99.7%
#      # pdf = probability distribution function
#      c1min = self.clusters[w_c_id].cluster['obb'].min
#      c2min = pc_cluster.cluster['obb'].min
#      c1max = self.clusters[w_c_id].cluster['obb'].max
#      c2max = pc_cluster.cluster['obb'].max
#      #
#      min_mean, min_stddev = self.clusters[w_c_id].get_mean_stddev('obb_attr',"MIN_DIF_STATS") 
#      max_mean, min_mean= self.clusters[w_c_id].get_mean_stddev('obb_attr',"MAX_DIF_STATS") 
#      for c1Xmax, c2Xmax in zip(c1max, c2max):
#        for c1Xmin, c2Xmin in zip(c1min, c2min):
#          # x, y, z
#          val = abs(c1Xmax - c2Xmax)
#          if min_mean != None and min_stddev != None:
#            bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(val)
#          else:
#            bb_min_prob = 0
#          val = abs(c1Xmin - c2Xmin)
#          if max_mean != None and max_stddev != None:
#            bb_max_prob = scipy.stats.norm(max_mean, max_stddev).pdf(val)
#            print("bb_max_prob:", bb_max_prob, bb_min_prob)
#          else:
#            bb_max_prob = 0
#          # print("bb_max_prob:", bb_max_prob, bb_min_prob)
#      loc_indep_prob = 1
#      # for i in range(3):
#      #   loc_indep_prob = min(loc_indep_prob, bb_min_prob[i], bb_max_prob[i])
#      loc_indep_prob = min(bb_min_prob, bb_max_prob)
#      loc_dep_prob = loc_indep_prob * loc_dep_prob
#      return loc_dep_prob, loc_indep_prob

    # associate grasp with the cluster
    def compare_location(self, c_id, pc_cluster):
      pass

#    # for integration into world cluster
#    # def compare_keypoints(self, c_id, pc_cluster):
#    def compare_keypoints_old(self, pc_cluster, KPs):
#      KPList = pc_cluster.cluster['kp'].get_kp()
#      print("KPList",KPList)
#      for kp_i, KP in enumerate(KPList):
#        # KP.get_kp()
#        # KP.get_descriptor()
#        # return list_cluster[0],score2
#
#        cluster_match, score = self.clusters[c_id].cluster['kp'].compare_kp(pc_cluster)
#        # just use ratios
#        # kp_mean, kp_stddev = self.get_mean_stddev(c_id,'kp_attr',"KP_STATS") 
#        # bb_min_prob = scipy.stats.norm(min_mean, min_stddev).pdf(kp_distance)
#        #
#        # ratio test as per Lowe's paper
#        # for i,(m,n) in enumerate(flann_matches):
#          # ratio[i] = m.distance / n.distance
#        #
#        # for i,(m,n) in enumerate(bf_matches):
#          # ratio[i] = m.distance / n.distance
#        #
#      # for i,(m,n) in enumerate(bf_matches):
#        # if m.distance < 0.7*n.distance:
#      # for i,(m,n) in enumerate(flann_matches):
#        # if m.distance < 0.7*n.distance:
#      # set_attr(self, c_id, field, key_value):
#      #
#      rot_mean, rot_stddev = self.clusters[c_id].get_mean_stddev('obb_attr',"ROTATION_STATS")
#      rot_val = pc_cluster['obb'].rotation
#      for r_v in zip(rot_val):
#        rot_prob = scipy.stats.norm(rot_mean, rot_stddev).pdf(r_v)
#      loc_dep_prob = 1
#      for i in range(3):
#        loc_dep_prob = min(loc_dep_prob, rot_prob[i], cent_prob[i])
#      # ARD: rotation???
#      return kp_prob 

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
    def in_any_obb(self, point):
        for c_id, c in enumerate(self.clusters):
          ret = c.in_bounding_box(point)
          if ret[0] == True:
            print("grasp in cluster",c_id)
            return True
        print("grasp not in cluster")
        return False


    def assign_grasps(self, grasp_conf):
      # there should be no movement since analyze_pc
  
      found = False
      g_unfound = []
      g_found = []
      if grasp_conf == None:
        return False
      for g_idx, (g, prob) in enumerate(grasp_conf):
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
         # ARD: bug?
         # cluster_center = c.cluster['obb'].centroid
         # obb = c.cluster['obb']
         # print("min", obb.get_min)
         # print("max", obb.get_max)
         # print("cnt", obb.centroid)
         # ret = c.in_bounding_box(cluster_center)
         # if ret == True:
         #   print("Success: Cluster center in bounding box")
         # else:
         #   print("Failure: Cluster center not in bounding box")
         # vert = obb.points[1]
         # ret = c.in_bounding_box(vert)
         # if ret == True:
         #   print("Success: vertex in bounding box")
         # else:
         #   print("Failure: vertex not in bounding box")

         ret = c.in_bounding_box([g[0],g[1],g[2]])
         if ret[0] == None or ret[0] == False:
            # cluster_center = c.cluster['centroid']
            cluster_center = c.cluster['obb'].centroid
            g_cent_dist = distance_3d(cluster_center, g)
            # print(" c_id ", c_id," centroid is ", cluster_center," vs dist ", g_cent_dist)
            if g_cent_dist < d1[1]:
              d3 = d2
              d2 = d1
              d1 = [c_id, g_cent_dist]
              # print("No OBB; closest to center: ", d1, d2, d3)
            elif g_cent_dist < d2:
              d3 = d2
              d2 = [c_id, g_cent_dist]
            elif g_cent_dist < d2:
              d3 = [c_id, g_cent_dist]
            continue
         else:
            # print("grip ",g, " in obb for ",c_id)
            # ARD TODO: transform to take flip/rotate into consideration
            c.cluster['num_grasps'] += 1
            c.cluster['grasps'].append(g)
            c.cluster['grasp_attr'] = []            # ["name",value] pair
            g_found.append([g_idx, c_id, ret[1]])
            found = True
         continue
        if not found:
          # print("no cluster found for grasp ", g)
          g_unfound.append([g_idx, d1])
          # print("best fits ", d1,d2,d3)
        continue
      # Return doesn't matter
      #
      # too many duplicate clusters is a bad sign of overlapping OBB
      prev_gidx = -1
      curcnt    = 0
      maxcnt    = 0
      numdup    = 0
      min_dif    = BIGNUM
      min_difcid = -1
      for i,g in enumerate(g_found):
        if g[0] != prev_gidx:
          prev_gidx = g[0]
          maxcnt = max(maxcnt, curcnt)
          if min_dif > g[2]:
            min_difcid = g[1]
            min_dif    = g[2]
          if curcnt > 1:
            numdup += 1
          curcnt = 0
        curcnt += 1
      maxcnt = max(maxcnt, curcnt) # get last one
      # Grips with multiple clusters: can clusters may be combined?
      # ARD: Rank by secondarily min distance to x/y center of same cluster?

      print("# Grips with known clusters/dups/maxcnt  :", len(g_found), numdup, maxcnt) 
      print("Grips with unknown clusters:", len(g_unfound))  
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

#     For full world KPs, compare to current full pc KPs
#     Fine matching KP infos, and compute distance moved
    def compare_all_keypoints(self, pc_clusters):
      self.kp_w_pc_info_match = compare_w_pc_kp(self, self.kp_pc_info, pc_clusters.kp_info)

#     For each world cluster, generate cluster 2d image
#     Compute and detect KPs for each cluster 2d image
#     Compare full set of new KPs to each world cluster KPs
#     Map new KPs to new clusters
#     Map new clusters to world clusters
#     Future: handle rotation, flipping, rolling, compare and combine clusters
    def compare_keypoints(self, KP):
      pc_kp_prob = []
      # print("KP:",KP.get_kp())
     
      pc_kp_matching_cluster = []
      pc_kp_distance = []
      # pc_kp_xy = []
      # pc_kp_3dpt = []
      pc_kp_obb = []
      pc_kp_pts = []
      for w_c_id, c in enumerate(self.clusters):
        pc_kp_matching_cluster.append(None)
        pc_kp_distance.append(None)
        pc_kp_pts.append(None)
        # pc_kp_obb.append(None)
      for i, c in enumerate(self.clusters):
        # get pointcloud associated with designated cluster
        cluster_pc = c.cluster['shape']
        print("len(cluster_pc):", len(cluster_pc))
        # Generate 2d image from 3d pointcloud
        rgb_pc = cluster_pc
        # print(cluster_pc[0])
        cluster_img, depth, pc_map, cluster_kp_img = rgb_depth_map_from_pc(cluster_pc, rgb_pc, fill=False)
        # return rgb, depth, pc_map, pc_img
        # cluster_img is x=RGB_WIDTH/y=RGB_HEIGHT/val=RGB

        # Compute and detect KPs from 2d image of single world cluster
        c.cluster['kp'] = Keypoints(cluster_img)
        print("num w  cluster KPs: ", i, len(c.cluster['kp'].get_kp()))
        print("num pc cluster KPs: ", len(KP.get_kp()))

        # list of [kp_pc3dpt, pc_c_id, pc[i]]
        kp_c_pc_mapping = c.cluster['kp_c_pc_mapping']
        # compare cluster's KPs to full set of KPs from latest image 
        # matching_pc_cluster[i], score[i], pc_kp_xy[i], pc_kp_3dpt[i]  = c.cluster['kp'].compare_kp(pc_c, KP, kp_c_pc_mapping)
        if c.cluster['kp'] != None:
          a1, b1, c1 = c.cluster['kp'].compare_cluster_kp(KP, kp_c_pc_mapping)
          pc_kp_matching_cluster[i] = a1
          pc_kp_distance[i] = b1
          pc_kp_pts[i] = c1
          # pc_kp_obb[i] = d1
        if pc_kp_matching_cluster[i] != None and len(pc_kp_matching_cluster[i]) > 0:
          print("w",i, "matching keypoint pc",pc_kp_matching_cluster[i]," distance ", pc_kp_distance[i])
      return pc_kp_matching_cluster, pc_kp_pts, pc_kp_distance 
      # return pc_kp_matching_cluster, pc_kp_distance, pc_kp_score, pc_kp_xy, pc_kp_3dpt, pc_kp_obb


#     Map new KPs to new clusters
#     Map new clusters to world clusters
#     compare and combine clusters
#     Future: handle rotation, flipping, rolling


    def publish_obb(self):
      markerArray = MarkerArray()
      marker = Marker()
      marker.header.frame_id = "/base_footprint"
      marker.type = marker.LINE_LIST
      marker.action = marker.ADD
  
      # marker scale
      # marker.scale.x = 0.03
      marker.scale.x = 0.001
      marker.scale.y = 0.001
      marker.scale.z = 0.001
  
      # "line width", 0.001,

      # marker color
      marker.color.a = 1.0
      marker.color.r = 1.0
      marker.color.g = 1.0
      marker.color.b = 0.0
  
      # marker orientaiton
      marker.pose.orientation.x = 0.0
      marker.pose.orientation.y = 0.0
      marker.pose.orientation.z = 0.0
      marker.pose.orientation.w = 1.0
  
      # marker position
      marker.pose.position.x = 0.0
      marker.pose.position.y = 0.0
      marker.pose.position.z = 0.0
  
      lines = [[0,1],[1,2],[2,3],[3,0],[0,5],[1,4],[2,7],[3,6],[4,5],[5,6],[6,7],[7,4]]
      marker.points = []
      id = 0
      # first point
      for c in self.clusters:
        obb = c.cluster['obb']
        linepts = obb.points
        for i in range(len(lines)):
          lp = Point()
          lp.x, lp.y, lp.z = linepts[lines[i][0]]
          marker.points.append(lp)
          lp = Point()
          lp.x, lp.y, lp.z = linepts[lines[i][1]]
          marker.points.append(lp)
          markerArray.markers.append(marker)
          # Renumber the marker IDs
          for m in markerArray.markers:
            m.id = id
            id += 1
  
      # Publish the Marker
      # self.obb_pub.publish(marker)
      self.obb_pub.publish(markerArray)
