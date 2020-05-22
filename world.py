#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
import collections 
from sensor_msgs.msg import (Image)
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import DBSCAN

from scipy import spatial
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
    # BOTH APIs : World and Latest PC
    ################################
    def __init__(self):
      self.state = 'UNINITIALIZED' 
      self.world_state = 'UNINITIALIZED' 
                        # ANALYZE PLAYGROUND, ANALYZE CLUSTERS, 
                        # ISOLATE, INTERACT
      self.num_clusters          = []
      self.clusters              = []
      self.octoclusters          = []
      self.octobase              = []
      self.possible_octobase     = []
      self.prev_target_w_cluster = None
      self.prev_target_w_grasp    = []
      self.prev_target_action    = []
      self.base_z                = None
      self.obb_prev_count        = 0
      self.obb_w_pc_pct_ovlp     = []
      self.w_expected_num_clusters = 0
      self.w_poss_base_pts       = []
      self.w_unmatched           = None
      self.w_best_match          = []
      self.w_unmoved_match       = None
      self.w_slightly_moved_match = None
      self.w_moved_match         = None
      self.w_active_list         = []
      self.pc_best_match         = []
      self.pc_unmatched          = None
      self.pc_unmoved_match      = None
      self.pc_slightly_moved_match = None
      self.pc_moved_match        = None
      self.pc_unmatched          = None

      self.pc_octocluster_pub = rospy.Publisher(PC_OCTOCLUSTER_TOPIC, PointCloud2, queue_size=1)
      self.pc_octobase_pub    = rospy.Publisher(PC_OCTOBASE_TOPIC, PointCloud2, queue_size=1)
      self.obb_pub            = rospy.Publisher(OBB_TOPIC, MarkerArray, queue_size=1)

    def create_cluster(self, id = None, status = None, centroid = None, 
               shape = None, shape_attr = None, bounding_box = None,
               num_locations = None, location = None, 
               num_grasps = None, grasps = None, grasp_attr = None,
               interaction = None, interaction_attr = None,
               state = None): 
      # self.cluster = {}  # done during init
      new_cluster = ClusterState()
      new_cluster.create_cluster(id, status,centroid, shape, shape_attr, bounding_box, num_locations, location, num_grasps, grasps, grasp_attr, interaction, interaction_attr, state)
      self.clusters.append(new_cluster)

    ###############################
    # WORLD APIs : persistent state
    ###############################

    def initialize_world(self, pc_clusters):
      # copy over initial analyzed pc_clusters
      self.state = 'ANALYZE PLAYGROUND'
      self.world_state = 'ANALYZE PLAYGROUND'
                        # ANALYZE PLAYGROUND, ANALYZE CLUSTERS,
                        # ISOLATE, INTERACT
      # self.clusters = pc_clusters.clusters
      self.w_best_match = []
      pc_clusters.pc_best_match = []
      self.w_expected_num_clusters = len(pc_clusters.clusters) - len(pc_clusters.possible_octobase)
      self.base_z = pc_clusters.base_z
      self.octoclusters = pc_clusters.octoclusters
      self.octobase = pc_clusters.octobase
      for pc_c_id,pc_c in enumerate(pc_clusters.clusters):
        if pc_c_id not in pc_clusters.possible_octobase:
          n = len(self.clusters)
          self.clusters.append(pc_c)
          self.clusters[n].id = n
          self.clusters[n].cluster['status'] = 'ACTIVE'
      for pc_c_id,pc_c in enumerate(pc_clusters.clusters):
        if pc_c_id in pc_clusters.possible_octobase:
          self.w_poss_base_pts.append(pc_c.cluster['shape'])

    def set_target_grasp(self, w_c_id, grasp, action):
      self.prev_target_w_cluster = w_c_id
      self.prev_target_w_grasp = grasp
      self.prev_target_action = action

    def active(self, w_c_id):
      if self.clusters[w_c_id].cluster['status'] == 'ACTIVE':
        return True
      return False
      
    # store in file
    def save_world(self):
      pass

    # read from file
    def retore_world(self):
      pass

      # we keep a sequence of pc state history
      # there is one integrated world state
      # the world state keeps a current
      #    cluster => ACTIVE, INACTIVE, POSSIBLE_BASE

      # do we need to reorder or just remap?
      # remapping requires multi level of remapping
      # but what if knowledge is gained over time?
      #    -> want knowledge to be accumulated to most recent...
      # Keep an array chain of pc_world_states? with mappings?
      # copy over latest state to world?
      # have separate accumulated state? (using copy)
      #    => base_z, octobase, possible base
      #    => state transistions
      #    => self.w_poss_base_pts = []
      #    => target
      # handle "has target cluster moved?"
      # some of the data gets stored

      # How state is used:
      #   expected_number_of_clusters
      #   try different objects (use history)
      #   knowledge that a piece is missing (inside a box)
      #   knowledge about hidden part of an object
      #   hard/soft
      #   knowledge of different grips success / fail
      #   "has target moved?"
      #   knowledge that two objects were stacked
      #   knowledge of base 
      #   knowledge of unstacking / stacking 
      #   knowledge of rolling
      #      -> good for making a goal / knocking over
      #   failed perfection match
      #
      # x,y,z calibration offset (not a world level?)
  


    def copy_world_state_to_curr_pc(self, w_clusters):
      # pc rederives from scratch
      self.w_expected_num_clusters = w_clusters.w_expected_num_clusters 
      self.w_best_match = w_clusters.w_best_match 
      self.obb_prev_count        = w_clusters.obb_prev_count
      self.obb_w_pc_pct_ovlp     = w_clusters.obb_w_pc_pct_ovlp

    def copy_curr_pc_to_world_state(self, pc_clusters):
      for pt in pc_clusters.octobase:
        if not pt_in_lst(pt, self.octobase):
          self.octobase.append(pt)
      # in world, possible_octobase
      for poss_ob in pc_clusters.possible_octobase:
        for pt in pc_clusters.clusters[poss_ob].cluster['shape']:
          if not pt_in_lst(pt, self.octobase) and not pt_in_lst(pt, self.w_poss_base_pts):
            self.w_poss_base_pts.append(pt)

      # compare over time
      print("copy over pc clusters")
      self.base_z                = pc_clusters.base_z
      # should evaluate how base has changed over time

      #########################
      # Move SPLIT/COMBINE to integrate?
      ####
      # COMBINE
      ####
      pc_poss_base = tuple(pc_c_id for pc_c_id in pc_clusters.possible_octobase)
      seen = {}
      w_sum = {}
      dup_match = {}
      for w_c_id, pc_c_id in enumerate(self.w_best_match):
        if pc_c_id == None:
          continue
        if pc_c_id in pc_poss_base:
          # skip if part of base?
          continue
        if pc_c_id not in seen:
          seen[pc_c_id] = [w_c_id]
          w_sum[pc_c_id] = 0
          if self.obb_w_pc_pct_ovlp[w_c_id][pc_c_id] != None:
            w_sum[pc_c_id] += self.obb_w_pc_pct_ovlp[w_c_id][pc_c_id]
        else:
          if len(seen[pc_c_id]) >= 1:
            seen[pc_c_id].append(w_c_id)
            if self.obb_w_pc_pct_ovlp[w_c_id][pc_c_id] != None:
              w_sum[pc_c_id] += self.obb_w_pc_pct_ovlp[w_c_id][pc_c_id]
        
      for pc_c_id in seen:
        if len(seen[pc_c_id]) > 1 and w_sum < 1:
          for w_c_id in seen[pc_c_id]:
            print("pretty confident that ", w_c_id, " is split of ", pc_c_id)
            if self.clusters[w_c_id].cluster['type'] == 'MULTI_OBJECT':
              self.clusters[w_c_id].cluster['status'] ='INACTIVE'
              for r_id, [r, w_id] in enumerate(self.clusters[w_c_id].cluster['relation']):
                if r == "IS_PART_OF":
                  del self.clusters[w_id].cluster['relation'][r_id]
                  self.clusters[w_id].cluster['status'] ='ACTIVE'
                  break
              print("pc", pc_c_id, " splits into w ", seen[pc_c_id])

      ####
      # SPLIT
      ####
      # multiple pc_c_id map to same w_c_id
      w_poss_base = tuple(w_c_id for w_c_id in self.possible_octobase)
      seen = {}
      pc_sum = {}
      dup_match = {}
      for pc_c_id, w_c_id in enumerate(self.pc_best_match):
        if w_c_id == None:
          continue
        if w_c_id in w_poss_base:
          continue
        if w_c_id not in seen:
          seen[w_c_id] = [pc_c_id]
          pc_sum[w_c_id] = 0
          if self.obb_pc_w_pct_ovlp[pc_c_id][w_c_id] != None:
            pc_sum[w_c_id] += self.obb_pc_w_pct_ovlp[pc_c_id][w_c_id]
        else:
          if len(seen[w_c_id]) >= 1:
            seen[w_c_id].append(pc_c_id)
            if self.obb_pc_w_pct_ovlp[pc_c_id][w_c_id] != None:
              pc_sum[w_c_id] += self.obb_pc_w_pct_ovlp[pc_c_id][w_c_id]
        
      for w_c_id in seen:
        if len(seen[w_c_id]) > 1 and pc_sum < 1:
          for pc_c_id in seen[w_c_id]:
            print("pretty confident that ", pc_c_id, " is combo of ", seen[w_c_id])
            pc_clusters.clusters[pc_c_id].cluster['status'] ='UNKNOWN'
            pc_clusters.clusters[pc_c_id].cluster['relation'].append(['IS_PART_OF', w_c_id])
          self.clusters[w_c_id].cluster['status'] ='ACTIVE'
          self.clusters[w_c_id].cluster['type'] ='MULTI_OBJECT'
          self.clusters[w_c_id].cluster['relation'].append(['CONSISTS_OF', seen[w_c_id]])

      #########################
      # copy over relevant pc state
      #########################

      for w_c_id, pc_bm in enumerate(self.w_best_match):
        if pc_bm == None:
          continue
        if pc_bm in pc_clusters.possible_octobase:
          continue
        # store the latest top-level pc_clusters info
        self.clusters[w_c_id].cluster['shape'] = pc_clusters.clusters[pc_bm].cluster['shape']
        self.clusters[w_c_id].cluster['obb'] = pc_clusters.clusters[pc_bm].cluster['obb']
        self.clusters[w_c_id].cluster['centroid'] = pc_clusters.clusters[pc_bm].cluster['centroid']
        self.clusters[w_c_id].cluster['center'] = pc_clusters.clusters[pc_bm].cluster['center']
        self.clusters[w_c_id].cluster['location'] = pc_clusters.clusters[pc_bm].cluster['location']
        flat_seg = self.clusters[w_c_id].find_attr('shape_attr','flat_pc')
        if flat_seg != None:
          self.clusters[w_c_id].set_attr('shape_attr','flat_pc',flat_seg)
      
        # TODO:
        # compute the attribute-level w_clusters info
        # cent_mean, cent_stddev = self.get_mean_stddev(c_id,'obb_attr',"CENTROID_STATS")
        # cent_val = pc_cluster['obb'].centroid
        # # if unmoved, computed mean and variance centroid
        # for c_v in zip(c1, c2):
        #   cent_prob = scipy.stats.norm(cent_mean, cent_stddev).pdf(c_v)
        # # rotation changes when moved
        # # if unmoved, computed mean and stddev centroid
        # rot_mean, rot_stddev = self.get_mean_stddev(c_id,'obb_attr',"ROTATION_STATS")
        # rot_val = pc_cluster['obb'].rotation

      for pc_um in self.pc_unmatched:
        if pc_um in pc_clusters.possible_octobase:
          continue
        # append new cluster
        n = len(self.clusters)
        clust = pc_clusters.clusters[pc_um]
        clust.id = len(self.clusters)
        clust.cluster['status'] = 'ACTIVE'
        clust.cluster['id'] = n
        self.clusters.append(clust)

      for wc_um in enumerate(self.w_unmatched):
        # mark cluster inactive
        self.clusters[wc_um].cluster['status'] = 'UNKNOWN'

      self.w_active_list = [w_c_id for w_c_id,w_c in enumerate(self.clusters) if w_c.cluster['status'] == 'ACTIVE']
      self.w_expected_num_clusters = 0
      self.save_world()

    # 2nd attempt to make a cluster based upon a subset of the octocluster.
    # The goal is to derive a cluster that is comparible to the world cluster.
    # unfortunately, the results do not seem promising. Better to skip for now
    # and do a comparison to the parked cluster. 
    def did_cluster_move(self, octomap, c_id, w_clusters):
        # compare pre-move octomap to post-grab octomap (with arm hovering above
        # cluster). Is this particular object there?
        # ARD: TODO
        w_obb = w_clusters.clusters[c_id].cluster['obb']
        w_center = w_clusters.clusters[c_id].cluster['center']
        w_cid_sz = len(w_clusters.clusters[c_id].cluster['shape'])
        # ARD DEBUG
        # approx_octomap = [pt for pt_cnt,pt in enumerate(octomap) if OBB.in_obb(w_obb, pt)[0] and pt_cnt < 5]
        # return False
        approx_octomap = [pt for pt in octomap if OBB.in_obb(w_obb, [pt[0],pt[1],pt[2]])[0]]
        approx_octoclusters_pc = [pt for pt in approx_octomap if w_clusters.base_z[get_sector(pt[0], pt[1])] > pt[2]]
        approx_octobase = [pt for pt in approx_octomap if w_clusters.base_z[get_sector(pt[0], pt[1])] <= pt[2]]
        print("did_cluster_mv:", len(approx_octoclusters_pc), len(approx_octobase), len(approx_octomap))
        octoclust = self.analyze_octoclusters(approx_octoclusters_pc, approx_octobase, approx_octomap)
        c_center = None
        for c_id, c in enumerate(octoclust):
          if abs(len(octoclust[c_id]) - w_cid_sz ) > .1 * w_cid_sz:
            continue
          counter = 0
          running_sum = np.array([0.0, 0.0, 0.0, 0.0])
          for pnt in octoclust[c_id]:
            counter += 1
            running_sum += pnt
          c_center = running_sum / counter
          if distance_3d(c_center, w_center) < 2.1 * OCTOMAP_RESOLUTION:
            print("cluster did not move", c_id, distance_3d(c_center, w_center))
            return False
        if c_center == None:
          print("no matching cluster. assuming moved.")
        else:
          print("cluster moved", c_id, distance_3d(c_center, w_center))
        return True

    # integrates temporary "pc cluster" into persistent "world" cluster
    # Now that the integration has morphed into pc->w and w->pc, the
    # following should become a procedure score_cluster(cl1,cl2) and
    # called for (pc,w) and (w,pc)
    def integrate_current_pc_into_world(self, pc_clusters):
      if self.world_state == 'UNINITIALIZED' or len(self.clusters) == 0:
        self.initialize_world(pc_clusters)
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
      # only one cluster can be closest
      centroid_matching_clusters = [None for w_c_id, w_c in enumerate(self.clusters)]
      centroid_min_dist = [BIGNUM for w_c_id, w_c in enumerate(self.clusters)]
      centroid_min_clust_dist = [BIGNUM for w_c_id, w_c in enumerate(self.clusters)]

      # multiple clusters combined or split...
      obb_pct_matching_clusters = [[] for i, w_c in enumerate(self.clusters)]
      obb_w_pc_pct_ovlp = [[] for i, w_c in enumerate(self.clusters)]
      obb_pc_w_pct_ovlp = [[] for i, pc_c in enumerate(pc_clusters.clusters)]
      obb_ovlp_cnt = [0 for i, w_c in enumerate(self.clusters)]
      obb_vol_match = [[] for i, w_c in enumerate(self.clusters)]

      pc_centroid_matching_clusters = [None for pc_c_id, pc_c in enumerate(pc_clusters.clusters)]
      pc_centroid_min_dist = [BIGNUM for pc_c_id, pc_c in enumerate(pc_clusters.clusters)]
      pc_centroid_min_clust_dist = [BIGNUM for pc_c_id, pc_c in enumerate(pc_clusters.clusters)]
      pc_obb_pct_matching_clusters = [[] for i, pc_c in enumerate(pc_clusters.clusters)]
      pc_obb_ovlp_cnt = [0 for i, pc_c in enumerate(pc_clusters.clusters)]
      pc_obb_vol_match = [[] for i, pc_c in enumerate(pc_clusters.clusters)]

      # Do compare of current PC to each cluster
      for pc_c_id, pc_c in enumerate(pc_clusters.clusters):
        # world
        for w_c_id, w_c in enumerate(self.clusters):
          if not self.active(w_c_id):
            continue
          ##############
          # Compare Centroid/Cluster distance
          ##############
          centroid_dist, clust_dist = self.cluster_approx_distance(w_c_id, pc_c)
          # print("centroid/clust dist:", centroid_dist, clust_dist )
          if centroid_min_dist[w_c_id] > centroid_dist:
            centroid_matching_clusters[w_c_id] = pc_c_id
            centroid_min_dist[w_c_id] = centroid_dist
            centroid_min_clust_dist[w_c_id] = clust_dist
          if pc_centroid_min_dist[pc_c_id] > centroid_dist:
            # pc_centroid_matching_clusters[pc_c_id] = pc_c_id
            pc_centroid_matching_clusters[pc_c_id] = w_c_id
            pc_centroid_min_dist[pc_c_id] = centroid_dist
            pc_centroid_min_clust_dist[pc_c_id] = clust_dist

          ##############
          # Compare OBB overlap based upon actual position
          #    - same object rotated?
          #    - same object moved slightly?
          ##############
          if clust_dist <= 0: 
            w_pc_obb_pct_ovlp, pc_w_obb_pct_ovlp = self.compare_bounding_box(w_c_id, pc_c)
            if w_pc_obb_pct_ovlp == None or pc_w_obb_pct_ovlp == None:
              obb_pct_matching_clusters[w_c_id].append(pc_c_id)
              obb_w_pc_pct_ovlp[w_c_id].append(None)
              obb_pc_w_pct_ovlp[pc_c_id].append(None)
              pc_obb_pct_matching_clusters[pc_c_id].append(w_c_id)
            else:
              obb_pct_matching_clusters[w_c_id].append(pc_c_id)
              obb_w_pc_pct_ovlp[w_c_id].append(w_pc_obb_pct_ovlp)
              obb_pc_w_pct_ovlp[pc_c_id].append(pc_w_obb_pct_ovlp)
              pc_obb_pct_matching_clusters[pc_c_id].append(w_c_id)

            # indicator that we should combine clusters
            if w_pc_obb_pct_ovlp > .70 or pc_w_obb_pct_ovlp > .70:
              obb_ovlp_cnt[w_c_id] += 1

            ##############
            # Compare OBB volumes (position independent)
            # to handle multiple of same objects, keep a list
            ##############
            w_obb  = self.clusters[w_c_id].cluster['obb']
            w_vol = OBB.obb_volume(w_obb)
            pc_obb = pc_clusters.clusters[pc_c_id].cluster['obb']
            pc_vol = OBB.obb_volume(pc_obb)
            if w_vol != None and pc_vol != None:
              obb_vol_match[w_c_id].append(w_vol / pc_vol)
              pc_obb_vol_match[pc_c_id].append(w_vol / pc_vol)
            else:
              # vol match is near 1, No match near 0.
              print("None w or pc obb vol:", w_vol, pc_vol)
              obb_vol_match[w_c_id].append(0)
              pc_obb_vol_match[pc_c_id].append(0)
          else:
              obb_pct_matching_clusters[w_c_id].append(pc_c_id)
              obb_w_pc_pct_ovlp[w_c_id].append(None)
              obb_pc_w_pct_ovlp[pc_c_id].append(None)
              pc_obb_pct_matching_clusters[pc_c_id].append(w_c_id)
              obb_vol_match[w_c_id].append(0)
              pc_obb_vol_match[pc_c_id].append(0)

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

      num_clust_diff = len(self.clusters) - len(pc_clusters.clusters)
      print("  # clust diff    = ", num_clust_diff)
      poss_base_clusters = len(self.possible_octobase)
      print("poss base clust1: ", poss_base_clusters)
      print("poss base clust2: ", len(pc_clusters.possible_octobase))
      PRINT_SUMMARY_DETAILS = False
      # need to clean up and move down
      if PRINT_SUMMARY_DETAILS:
        # prints out a ton (w_cl x pc_cl)
        print("  centroid_matching_clusters: ", centroid_matching_clusters)
        print("  centroid_min_dist         : ", centroid_min_dist)
        print("  centroid_min_clust_dist   : ", centroid_min_clust_dist)
        print("  obb_pct_matching clusters : ", obb_pct_matching_clusters)
        print("  obb_w_pc_pct_ovrlp        : ", obb_w_pc_pct_ovlp)
        print("  obb_pc_w_pct_ovrlp        : ", obb_pc_w_pct_ovlp)
        print("  obb_vol_match             : ", obb_vol_match)

      #####################################
      # Validation 
      # verification checks: are the matches consistent across the tests and clusters
      #####################################
      w_match  = [[] for i, w_c in enumerate(self.clusters)]
      w_score    = [[] for i,w_c in enumerate(self.clusters)]
      w_test   = [[] for i,w_c in enumerate(self.clusters)]
      w_best_match  = [None for i, w_c in enumerate(self.clusters)]

      pc_match = [[] for i, pc_c in enumerate(pc_clusters.clusters)]
      pc_score    = [[] for i,w_c in enumerate(pc_clusters.clusters)]
      pc_test   = [[] for i,w_c in enumerate(pc_clusters.clusters)]
      pc_best_match  = [None for i, w_c in enumerate(pc_clusters.clusters)]

      ERR_MARGIN = 0.1
      ##############
      # Unmoved Clusters
      ##############
      diff_cnt = len(self.clusters) - len(pc_clusters.clusters)

      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        # whether same cluster, or if moved
        cent_match      = centroid_matching_clusters[w_c_id]
        cent_dist       = centroid_min_dist[w_c_id]
        cent_clust_dist = centroid_min_clust_dist[w_c_id]
        score_cent = 0 
        if cent_dist < 2.1 * OCTOMAP_RESOLUTION and cent_clust_dist < 0: 
          score_cent = 100 
          w_test[w_c_id].append('cent')
        elif cent_clust_dist < 0: 
          # potential minor move
          w_test[w_c_id].append('cent_dist')
          score_cent = 75 / abs(1 + cent_dist)
        # else: potentially a major move; identified later
        if score_cent > 0:
          w_score[w_c_id].append(score_cent)
          w_match[w_c_id].append(cent_match)

      for pc_c_id, pc_c in enumerate(pc_clusters.clusters):
        # whether same cluster, or if moved
        pc_cent_match      = pc_centroid_matching_clusters[pc_c_id]
        pc_cent_dist       = pc_centroid_min_dist[pc_c_id]
        pc_cent_clust_dist = pc_centroid_min_clust_dist[pc_c_id]
        score_cent = 0
        if pc_cent_dist < 2.1 * OCTOMAP_RESOLUTION and pc_cent_clust_dist < 0:
          pc_score_cent = 100
          pc_test[pc_c_id].append('pc_cent')
        elif pc_cent_clust_dist < 0:
          # potential minor move
          pc_test[pc_c_id].append('pc_cent_dist')
          pc_score_cent = 75 / (1 + pc_cent_dist)
        # else: potentially a major move; identified later
        if pc_score_cent > 0:
          pc_score[pc_c_id].append(pc_score_cent)
          pc_match[pc_c_id].append(pc_cent_match)


      ##############
      # OBB with same volume 
      # (location independent)
      ##############
      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        if w_c_id in self.possible_octobase:
          continue
        same_vol = []
        for pc_c_id in obb_pct_matching_clusters[w_c_id]:
          if abs(obb_vol_match[w_c_id][pc_c_id] - 1) < ERR_MARGIN:
            same_vol.append(pc_c_id)
        for i in range(len(same_vol)):
          # 100 split among matches
          w_test[w_c_id].append('obb_vol')
          w_score[w_c_id].append(100 / len(same_vol))
          w_match[w_c_id].append(same_vol[i])

      for pc_c_id, pc_c in enumerate(pc_clusters.clusters):
        if pc_c_id in self.possible_octobase:
          continue
        same_vol = []
        for w_c_id in pc_obb_pct_matching_clusters[pc_c_id]:
          if not self.active(w_c_id):
            continue
          if abs(obb_vol_match[w_c_id][pc_c_id] - 1) < ERR_MARGIN:
            same_vol.append(w_c_id)
        for i in range(len(same_vol)):
          # 100 split among matches
          pc_test[pc_c_id].append('obb_vol')
          pc_score[pc_c_id].append(100 / len(same_vol))
          pc_match[pc_c_id].append(same_vol[i])

      ##############
      # OBB that subsumes another
      # (location dependent)
      ##############
      # One OBB is consumed by another (cleanup of clusters)
      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        if w_c_id in self.possible_octobase:
          continue
        for pc_offset, pc_c_id in enumerate(obb_pct_matching_clusters[w_c_id]):
          # print("w_c_id,pc_c_id, len obb_pct_match:", w_c_id, pc_c_id, len(obb_pct_matching_clusters), len(obb_pct_matching_clusters[w_c_id]))
          if obb_w_pc_pct_ovlp[w_c_id][pc_c_id] == None or obb_pc_w_pct_ovlp[pc_c_id][w_c_id] == None:
            w_test[w_c_id].append('obb_pct_none')
            w_score[w_c_id].append(0)
            w_match[w_c_id].append(pc_c_id)
          elif abs(obb_w_pc_pct_ovlp[w_c_id][pc_c_id] - 1) < ERR_MARGIN:
            w_test[w_c_id].append('obb_pct1')
            w_score[w_c_id].append(100)
            w_match[w_c_id].append(pc_c_id)
          elif abs(obb_pc_w_pct_ovlp[pc_c_id][w_c_id] - 1) < ERR_MARGIN:
            w_test[w_c_id].append('obb_pct2')
            w_score[w_c_id].append(100)
            w_match[w_c_id].append(pc_c_id)
          elif (obb_w_pc_pct_ovlp[w_c_id][pc_c_id] > 0 and
                obb_pc_w_pct_ovlp[pc_c_id][w_c_id] > 0 ):
            w_test[w_c_id].append('obb_pct3')
            w_score[w_c_id].append(50 * min(1,max(obb_w_pc_pct_ovlp[w_c_id][pc_c_id], obb_pc_w_pct_ovlp[pc_c_id][w_c_id])))
            w_match[w_c_id].append(pc_c_id)
            if obb_w_pc_pct_ovlp[w_c_id][pc_c_id] > 1 or obb_pc_w_pct_ovlp[pc_c_id][w_c_id] > 1:
              print("obb_w_pc_pct_ovlp, pc_w: ", obb_w_pc_pct_ovlp[w_c_id][pc_c_id], obb_pc_w_pct_ovlp[pc_c_id][w_c_id], w_c_id, pc_c_id)
          else:
            w_test[w_c_id].append('obb_pct_zero')
            w_score[w_c_id].append(0)
            w_match[w_c_id].append(pc_c_id)

      for pc_c_id, pc_c in enumerate(pc_clusters.clusters):
        if pc_c_id in pc_clusters.possible_octobase:
          continue
        for pc_offset, w_c_id in enumerate(pc_obb_pct_matching_clusters[pc_c_id]):
          # print("w_c_id,pc_c_id, len pc_obb_pct_match:", w_c_id, pc_c_id, len(pc_obb_pct_matching_clusters), len(pc_obb_pct_matching_clusters[pc_c_id]))
          if obb_w_pc_pct_ovlp[w_c_id][pc_c_id] == None or obb_pc_w_pct_ovlp[pc_c_id][w_c_id] == None:
            pc_test[pc_c_id].append('pc_obb_pct_none')
            pc_score[pc_c_id].append(0)
            pc_match[pc_c_id].append(w_c_id)
          elif (obb_w_pc_pct_ovlp[w_c_id][pc_c_id] - 1) < ERR_MARGIN:
            pc_test[pc_c_id].append('pc_obb_pct1')
            pc_score[pc_c_id].append(100)
            pc_match[pc_c_id].append(w_c_id)
          elif (obb_pc_w_pct_ovlp[pc_c_id][w_c_id] - 1) < ERR_MARGIN:
            pc_test[pc_c_id].append('pc_obb_pct2')
            pc_score[pc_c_id].append(100)
            pc_match[pc_c_id].append(w_c_id)
          elif (obb_w_pc_pct_ovlp[w_c_id][pc_c_id] > 0 and
                obb_pc_w_pct_ovlp[pc_c_id][w_c_id] > 0 ):
            pc_test[pc_c_id].append('pc_obb_pct3')
            pc_score[pc_c_id].append(50 * min(1, max(obb_w_pc_pct_ovlp[w_c_id][pc_c_id], obb_pc_w_pct_ovlp[pc_c_id][w_c_id])))
            pc_match[pc_c_id].append(w_c_id)
          else:
            pc_test[pc_c_id].append('pc_obb_pct_zero')
            pc_score[pc_c_id].append(0)
            pc_match[pc_c_id].append(w_c_id)

      ##############
      # Evaluate unmoved or slightly moved clusters
      ##############
      # Note: if w_c_id and pc_c_id match the above tests, we're pretty certain
      # these are the same objects.
      unmoved_match = []
      moved_match = []
      slightly_moved_match = []
      best_match_cnt = 0
      poss_base_cnt  = 0
      poss_base = []
      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        if w_c_id in self.possible_octobase:
          poss_base.append(w_c_id)
          continue
        if len(w_score[w_c_id]) >= 3 and w_score[w_c_id][0] == 100 and w_score[w_c_id][1] == 100 and w_score[w_c_id][2] == 100:
          w_best_match[w_c_id] = w_match[w_c_id][0]
          if not w_best_match[w_c_id] in pc_clusters.possible_octobase:
            print("unmoved match:", w_c_id, w_match[w_c_id][0])
            unmoved_match.append(w_c_id) # pc side is w_match[w_c_id]
          else:
            poss_base.append(w_c_id)
        elif len(w_score[w_c_id]) >= 2:
          potential_pc       = [pc_id for pc_id in set(w_match[w_c_id])]
          potential_pc_score = [0     for pc_id in potential_pc] 
          # print("w_match", w_c_id, len(w_match[w_c_id]), w_match[w_c_id])
          # print("w_score", w_c_id, len(w_score[w_c_id]), w_score[w_c_id])
          for pc_idx,pc_id in enumerate(potential_pc):
            for s_i, s in enumerate(w_score[w_c_id]):
              if w_match[w_c_id][s_i] == pc_id:
                potential_pc_score[pc_idx] += s 
          best_score  = 0
          best_pc_idx = None
          for i in range(len(potential_pc_score)):
            if best_score < potential_pc_score[i]:
              best_score = potential_pc_score[i]
              best_pc_idx = i
          if best_pc_idx != None:
            w_best_match[w_c_id] = potential_pc[best_pc_idx]
            slightly_moved_match.append(w_c_id)
            if not w_best_match[w_c_id] in pc_clusters.possible_octobase:
              print("w best match:", w_c_id, w_best_match[w_c_id], best_score)
              best_match_cnt += 1
            else:
              poss_base_cnt += 1
      print("Unmoved clusters:")
      print("  unmoved_match       : ", unmoved_match)
      print("  slightly_moved_match: ", slightly_moved_match)
      print("  best_match_cnt      : ", best_match_cnt)
      print("  poss_base           : ", poss_base)

      ##############
      # Note: if w_c_id and pc_c_id match the above tests, we're pretty certain
      # these are the same objects.
      pc_unmoved_match = []
      pc_slightly_moved_match = []
      pc_best_match_cnt = 0
      pc_poss_base_cnt = 0
      for pc_c_id, pc_c in enumerate(pc_clusters.clusters):
        if pc_c_id in pc_clusters.possible_octobase:
          pc_poss_base_cnt += 1
          continue
        if len(pc_score[pc_c_id]) >= 3 and pc_score[pc_c_id][0] == 100 and pc_score[pc_c_id][1] == 100 and pc_score[pc_c_id][2] == 100:
          pc_best_match[pc_c_id] = pc_match[pc_c_id][0]
          if not pc_best_match[pc_c_id] in pc_clusters.possible_octobase:
            print("pc_perfect match:", pc_c_id, pc_match[pc_c_id][0])
            pc_unmoved_match.append(pc_c_id)
          else:
            pc_poss_base_cnt += 1
        elif len(pc_score[pc_c_id]) >= 2:
          potential_w       = []
          potential_w_score = []
          for w_idx,w_id in enumerate(set(pc_match[pc_c_id])):
            potential_w.append(w_id)
            potential_w_score.append(0)
            for s in range(len(pc_score[pc_c_id])):
              potential_pc_score[pc_idx] += s 
          best_score  = 0
          best_w_idx = 0
          for i in range(len(potential_w_score)):
            if best_score < potential_w_score[i]:
              best_score = potential_w_score[i]
              best_w_idx = i
          pc_best_match[pc_c_id] = potential_w[best_w_idx]
          if not pc_best_match[pc_c_id] in self.possible_octobase:
            print("pc best match:", pc_c_id, pc_best_match[pc_c_id], best_score)
            pc_best_match_cnt += 1
            pc_slightly_moved_match.append(pc_c_id)
          else:
            pc_poss_base_cnt += 1
      print("Unmoved clusters:")
      print("  pc_unmoved_match       : ", pc_unmoved_match)
      print("  pc_slightly_moved_match: ", pc_slightly_moved_match)
      print("  pc_best_match_cnt      : ", pc_best_match_cnt)
      print("  pc_poss_base_cnt       : ", pc_poss_base_cnt)
###############
# Need to copy over non-base perfect and best clusters to world state.
#          -> moved and unmatched clusters computed below
#
# clean up pc possible base and store in world state?
#          -> used during matching?
#          -> on non-base obb in world state?
#          -> possible_base is full resolution, unlike octobase subset
# 
# keep base_z + possible bases?
# what is needed to clean up grasps?
# 
# Since octobase can't be used as an optimization next frame, don't need to
# compute the rest?
###############
      ##############
      # Combine octobases
      ##############
      pre_len = len(self.octobase)
      # self.octobase = self.octobase + list(pc_clusters.octobase - self.octobase)
      # ARD: takes too long
      # self.octobase = np.concatenate((self.octobase, np.array([p for p in pc_clusters.octobase if not pt_in_lst(p, self.octobase)])))
      # tuple_of_tuples = tuple(tuple(x) for x in list_of_lists)
      # a = set(list(pc_clusters.octobase))
      pc_base_tuples = tuple(tuple(pt) for pt in pc_clusters.octobase)
      w_base_tuples  = tuple(tuple(pt) for pt in self.octobase)
      set_dif = set(pc_base_tuples).difference(w_base_tuples)
      self.octobase = w_base_tuples + tuple(set_dif)
      # self.octobase = np.concatenate((self.octobase, np.array(set(list(pc_clusters.octobase)).difference(list(self.octobase)))))
      if len(self.octobase) > pre_len:
        print("Points added to octobase:", (len(self.octobase) - pre_len))

      ##############
      # Combine possible pc & master octobases
      ##############
      prev_match_w = -1
      for w_c_id in self.possible_octobase:
        # combine w_c_id with overlapping pc_c_id that are possible base
        combined = False
        for pc_c_id in pc_clusters.possible_octobase:
          if pc_c_id in obb_pct_matching_clusters[w_c_id]:
            for pt in pc_clusters.clusters[pc_c_id].cluster['shape']:
              if pt not in self.clusters[w_c_id].cluster['shape']:
                self.clusters[w_c_id].cluster['shape'].append(pt)
            self.clusters[w_c_id].cluster['status'] = "BASE"
            pc_clusters.clusters[pc_c_id].cluster['shape'] = []
            pc_clusters.clusters[pc_c_id].cluster['status'] = "EMPTY"
            w_best_match[w_c_id] = pc_c_id
            print("combined potential bases: ", w_c_id, pc_c_id, len(self.clusters[w_c_id].cluster['shape']))
            prev_match_w = w_c_id
            combined = True
#        if len(self.octobase) > pre_len:
#          if combined:
#            self.clusters[w_c_id].compute_bounding_box()
          # combine w_c_id with other overlapping world clusters that are possible base
        combined2 = False
        for w_c_id2 in self.possible_octobase:
          if w_c_id != w_c_id2:
            obb1 = self.clusters[w_c_id].cluster['obb']
            obb2 = self.clusters[w_c_id2].cluster['obb']
            if obb1 != None and obb2 != None:
              ovlp1, ovlp2 = self.get_pct_overlap(obb1, obb2)
              if ovlp1 != None and ovlp2 != None:
                for pt in self.clusters[w_c_id2].cluster['shape']:
                  if pt not in self.clusters[w_c_id].cluster['shape']:
                    self.clusters[w_c_id].cluster['shape'].append(pt)
                # self.clusters[w_c_id].cluster['shape'].append(self.clusters[w_c_id2].cluster['shape'])
                self.clusters[w_c_id].cluster['status'] = "BASE"
                self.clusters[w_c_id2].cluster['shape'] = []
                self.clusters[w_c_id2].cluster['status'] = "EMPTY"
                print("combined potential w_bases: ", w_c_id, w_c_id2)
                w_best_match[w_c_id2] = None
                combined2 = True
        # both w_c_id and octobase may have changed. Remove duplicates from w_c_id.
        if combined or combined2:
          self.clusters[w_c_id].compute_bounding_box()

      ##############
      # successful pick/drop
      ##############
      print("prev target info:", self.prev_target_w_cluster, self.prev_target_w_grasp, self.prev_target_action)

      ##############
      # Likely cluster big move (push or unsuccessful pickup/drop)
      ##############
      # look for clusters with same volume.
      # may have multiple clusters of the same shape & size.
      min_dist = None
      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        if w_best_match[w_c_id] != None:
          # already know that the cluster is unmoved
          continue
        # pc_c_id's with same volumes (previously computed)
        for t_offset, t in enumerate(w_test[w_c_id]):
          if t == 'obb_vol' and w_score[w_c_id][t_offset] == 100:
            same_pc_vol = w_match[w_c_id][t_offset] 
        if len(same_pc_vol) == 1:
          # degenerate case. 
          w_best_match[w_c_id] = same_pc_vol[0]
          moved_match.append(w_c_id)
          pc_c = pc_clusters.clusters[same_pc_vol[pc_id]]
          centroid_dist, clust_dist = self.cluster_approx_distance(w_c_id, pc_c)
          min_dist = centroid_dist
        else:
          # assign by distance
          min_dist    = BIGNUM
          min_dist_pc = None
          for pc_id in same_pc_vol: 
            pc_c = pc_clusters.clusters[same_pc_vol[pc_id]]
            centroid_dist, clust_dist = self.cluster_approx_distance(w_c_id, pc_c)
            if min_dist < centroid_dist:
              min_dist = centroid_dist
              min_dist_pc = pc_id
          w_best_match[w_c_id] = min_dist_pc
          moved_match.append(min_dist_pc)

      print("--------------")
      print("w_best_match: ", w_best_match)
      print("w_dist moved: ", min_dist)

      # ARD: a while to next printout 


      ##############
      # Unmatched Clusters
      ##############
      # w_unmatched should be removed from self.clusters and appended to self.unmatched
      # pc_unmatched should be copied over to the world and added to self.clusters

      # How many w/pc are now unmatched?
      pc_poss_base_tuples  = tuple(pc_c_id for pc_c_id in pc_clusters.possible_octobase)
      pc_clust_tuples      = tuple(pc_c_id for pc_c_id,pc_c in enumerate(pc_clusters.clusters))
      pc_clust_tuples2     = tuple(pc_c_id for pc_c_id,pc_c in enumerate(pc_clusters.clusters) if pc_best_match[pc_c_id] != None)
      pc_best_match_tuples = tuple(pc_c_id for pc_c_id in w_best_match if pc_c_id != None)

      # w_unmatched = [w_c_id for w_c_id, w_c in enumerate(self.clusters) if w_best_match[w_c_id] == None and w_c_id not in self.possible_octobase and self.active(w_c_id)]
      w_poss_base_tuples  = tuple(w_c_id for w_c_id in self.possible_octobase)
      w_unmatch_tuples  = tuple(w_c_id for w_c_id,pc_c_id in enumerate(w_best_match) if w_best_match[w_c_id] == None and self.active(w_c_id))
      w_unmatched  = set(w_unmatch_tuples).difference(w_poss_base_tuples)

      # pc_unmatched = [pc_c_id for pc_c_id, pc_c in enumerate(pc_clusters.clusters) if pc_c_id not in w_best_match[:] and pc_c_id not in pc_clusters.possible_octobase]
      pc_unmatched = set(pc_clust_tuples).difference(pc_clust_tuples).difference(pc_poss_base_tuples)

      # pc_unmatched2 = [pc_c_id for pc_c_id, pc_c in enumerate(pc_clusters.clusters) if pc_best_match[pc_c_id] and pc_c_id not in pc_clusters.possible_octobase]
      pc_unmatched2 = set(pc_clust_tuples2).difference(pc_poss_base_tuples)
      print("len of w_unmatched, pc_unmatched, pc_unmatched2:", len(w_unmatched),len(pc_unmatched), len(pc_unmatched2))

      if len(w_unmatched) == 1 and len(pc_unmatched) == 1:
        w_best_match[w_unmatched[0]] = pc_unmatched[0]
        moved_match.append(w_unmatched[0])
        w_unmatched = []
        pc_unmatched = []
      elif len(w_unmatched) >= 1 and len(pc_unmatched) >= 1:
        w_info = []
        for w_c_id in w_unmatched:
          w_obb = self.clusters[w_c_id].cluster['obb']
          w_info.append([w_c_id, OBB.obb_volume(w_obb), w_obb.centroid])
        pc_info = []
        for pc_c_id in pc_unmatched:
          pc_obb = pc_clusters.clusters[pc_c_id].cluster['obb']
          pc_info.append([pc_c_id, OBB.obb_volume(pc_obb), pc_obb_centroid])
        pc_info2 = []
        for pc_c_id in pc_unmatched2:
          pc_obb = pc_clusters.clusters[pc_c_id].cluster['obb']
          pc_info2.append([pc_c_id, OBB.obb_volume(pc_obb), pc_obb_centroid])
        print("--------------")
        print("w_unmatched : ", w_info)
        print("pc_unmatched: ", pc_info)
        print("pc2unmatched: ", pc_info2)

      ##############
      # possible combined octoclusters
      ##############
      # If another pc cluster is missing, and 2 clusters together are within bounds,

      ##############
      # possible split octoclusters
      ##############
      # If another pc cluster is new, and two clusters together are within bounds,
      #    then possible split.

      ##############
      # cluster disappeared (put in a box? hidden? dropped off tray? 
      # now part of combined cluster?)
      ##############

      ##############
      # new cluster appeared (taken from box? unhidden? dropped onto tray?
      # no longer part of combine cluster?)
      ##############

      ##############
      # Complete integration
      ##############
      # do an initial scan of new frame assuming most objects didn't move.
      # always do the full computation for target clusters. 
      # Gather results when targeted (point, cluster), pushed, picked up, etc
      # whether it rolled. whether it moved when not a target.
      # store different OBB shapes.
      # use keypoints / color?
      # do close-up analysis (e.g., for perfection game)?
      # self.clusters only contains currently "active" clusters
      # self.old_clusters contains missing/unknown clusters
      self.w_best_match = []
      for w_c_id, w_c in enumerate(self.clusters):
        if not self.active(w_c_id):
            continue
        if (not w_c_id in self.possible_octobase and
            not w_best_match[w_c_id] in pc_clusters.possible_octobase):
          pc_clusters.w_best_match.append(w_best_match[w_c_id])
          self.w_best_match.append(w_best_match[w_c_id])
        else:
          pc_clusters.w_best_match.append(None)
          self.w_best_match.append(None)
      self.w_unmatched = w_unmatched
      self.w_unmoved_match = unmoved_match
      self.w_slightly_moved_match = slightly_moved_match 
      self.pc_unmatched = pc_unmatched
      self.obb_w_pc_pct_ovlp = obb_w_pc_pct_ovlp
      pc_clusters.pc_unmoved_match = pc_unmoved_match
      pc_clusters.pc_slightly_moved_match = pc_clusters.pc_slightly_moved_match 
      pc_clusters.w_unmatched = w_unmatched
      pc_clusters.pc_unmatched = pc_unmatched
      pc_clusters.obb_w_pc_pct_ovlp = obb_w_pc_pct_ovlp
      self.copy_curr_pc_to_world_state(pc_clusters)
      self.save_world()

#      print("Unmoved clusters:")
#      print("  unmoved_match_cnt: ", unmoved_match_cnt)
#      print("  best_match_cnt   : ", best_match_cnt)
#      print("  poss_base_cnt    : ", poss_base_cnt)
#      print("Points added to octobase:", (len(self.octobase) - pre_len))
#
#      print("combined potential bases: ", len(self.clusters[w_c_id].cluster['shape']))
#      print("prev target info:", self.prev_target_w_cluster, self.prev_target_w_grasp, self.prev_target_action)
#
#      print("w_unmatched : ", w_info)
#      print("pc_unmatched: ", pc_info)
#
#      print("w_best_match: ", w_best_match)
#      print("w_dist moved: ", min_dist)


#################################################################################


      # We maintain a short history of clusters:
      # world (integrated with all the previous parked images)
      #   -> understands real objects
      # Previous parked 
      # pre-grip -> compare to post-grip
      # gripper down -> compare target to end of gripper
      #              -> how to adjust gripper to get to target?
      #              -> SKIP FOR NOW
      # post-grip -> did target move? 
      # pre-drop -> compare to post drop???
      # post-drop
      # Next parked (pc, integrated)


################ END integration

    def octomap_score(self, curr_octomap_pc, goal_octomap_pc):
      # based upon pc occupancy difference
      # difference = (A - B) U (B - A)  where (A-B) is based upon occupancy 
      A_minus_B = len(goal_octomap_pc - curr_octomap_pc)
      B_minus_A = len(curr_octomap_pc - goal_octomap_pc)
      reward = len(goal_octomap_pc) + len(curr_octomap) - A_minus_B - B_minus_A
      return reward
      

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



    ###############################
    # LATEST PC APIs
    ###############################

    def publish_octo_pc(self, octoclusters, octobase, header, analyzed = False):
      if not DISPLAY_PC_CLUSTERS:
        return
      for i in range(2):
        if i == 0:
          print("publish octoclusters:",len(octoclusters))
          if analyzed:
            cluster_pc = []
            for c in octoclusters:
              for p in c:
                cluster_pc.append(p)
          else:
            cluster_pc = octoclusters
        else:
          if len(octobase) == 0:
            return
          cluster_pc = octobase
          # print("octobase")
          # cluster_pc = list(octobase)
          # print("len octobase", len(octobase))
        # cluster_pc = np.reshape(cluster_pc, (len(cluster_pc), 4))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        cluster_pc = point_cloud2.create_cloud(header, fields, cluster_pc)
        if i == 0:
          self.pc_octocluster_pub.publish(cluster_pc)
        else:
          self.pc_octobase_pub.publish(cluster_pc)


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

    #####################################################
    # find_octomap_clusters(self, pc):
    # This python plane segmentation is derived from:
    # https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/geometry/models/plane.py
    # https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/ransac/fitters.py
    #####################################################
    def segment_octocluster_pc(self, pc, base = []):
      from sklearn import linear_model
      from sklearn.metrics import r2_score, mean_squared_error
      cluster_pc = np.array(pc)[:, :3]
      # cluster_pc = np.random.shuffle(cluster_pc)
      points = cluster_pc
      # max_iterations=100
      # max_iterations= int(.2 * len(pc))
      max_iterations= int(len(pc))
      best_inliers = None
      n_inliers_to_stop = len(points)
      self.point = np.mean(points, axis=0)
      # print("mean ", self.point)
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
      # max_dist = .03
      # min_dist = 1.1 * get_approx_octmap_density(len(pc))
      # diag_nn_2d = sqrt(min_dist * min_dist * 2)
      # max_dist = 20*sqrt(diag_nn_2d * diag_nn_2d + min_dist * min_dist)
      # max_dist = 1.1 * get_approx_octmap_density(len(pc))
      max_dist = 1.1 * OCTOMAP_RESOLUTION
      # print("max_dist",max_dist, n_inliers_to_stop)
      print_once = True
      for i in range(max_iterations):
          if ((max_iterations - i) % 1000 == 0):
            print((max_iterations - i)) 
          # index = np.random.randint(0, len(pc)-4)
          # k_points = np.array(cluster_pc)[index:index+3, :3]

          sample = np.random.choice(len(cluster_pc), 3, replace=False)
          k_points = cluster_pc[sample]
          # print("sample", sample)
          # print("points", k_points)

          # normal = np.cross(points[1] - points[0], points[2] - points[0])
          p1 = k_points[1] - k_points[0]
          p2 = k_points[2] - k_points[0]
          # print("p1,2 shape: ", p1.shape, p2.shape, p1)
          normal = np.cross(p1,p2)
          self.point = k_points[0]
          if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            if print_once:
              print_once = False
              # print("normal: ",normal)
            # ARD: self.normal = [1,1,1]
            self.normal = [0,0,0]
          else:
            self.normal = normal / np.linalg.norm(normal)
          vectors = points - self.point
          all_distances = np.abs(np.dot(vectors, self.normal))
          inliers = all_distances <= max_dist
          n_inliers = np.sum(inliers)
          # print("all_distances:", len(all_distances), n_inliers, n_best_inliers, all_distances)
          if n_inliers > n_best_inliers:
              n_best_inliers = n_inliers
              best_inliers = inliers
              if n_best_inliers > n_inliers_to_stop:
                  break
      # print("plane: ", best_inliers)            # true/false array
      # print("len plane: ", len(best_inliers))
      # print("len pc   : ", len(pc))             # same len as above
      octobaselst = []
      octoclusterslst = []
      for i in range(len(best_inliers)):
        if best_inliers[i]:
          octobaselst.append(pc[i])
        else:
          octoclusterslst.append(pc[i])

      octobase = np.array(octobaselst)[:, :4]
      if len(base) > 0:
        # print("b4 concat:", np.shape(octobase))
        octobase = np.concatenate((octobase, base))
        # print("af concat:", np.shape(octobase))
      # print("octobase shape", octobase.shape)
      if len(octoclusterslst) > 0:
        octoclusters = np.array(octoclusterslst)[:, :4]
        # print("octoclusters shape", octoclusters.shape)
      else:
        # print("octoclusters len", len(octoclusterslst))
        octoclusters = []
      return octoclusters, octobase

    def analyze_octoclusters(self, octoclusters_pc, octobase, octomap):
        # find the individual objects in the combined octocluster pcs
        # octoclusters_pc is an approximation of a cluster based on
        # a small subset of points. 
        # octomap is the full octomap.
        # return a set of clusters based upon the full octomap details.
        def take_z_axis(elem):
          return elem[2]

        x,y,z = 0,1,2
        # pc is cluster only; tray has been filtered out
        # Start with the highest points, and compute neighbors in cluster.
        # Sorting used by cluster analysis to compute distances
        cluster_pc_rgb = sorted(octoclusters_pc, key=take_z_axis)
        cluster_pc = np.array(cluster_pc_rgb)[:, :3]
        cluster_pc_todo = [i for i in range(len(cluster_pc))]
        octomap_no_rgb = np.array(octomap)[:, :3]
        USE_2D_CL_RADIUS = False
        if USE_2D_CL_RADIUS:
          cluster_pc_2d = np.array(cluster_pc_rgb)[:, :2]

        ##################
        # CLUSTER ANALYSIS
        ##################
        sys.setrecursionlimit(20000)
        # kdtree       = spatial.KDTree(cluster_pc)
        kdtree       = spatial.KDTree(octomap_no_rgb)
        processed_octomap_pt = [False for i in range(len(octomap_no_rgb))]
        clusters     = []     
        # dynamically compute based upon pc size
        min_dist = 1.1 * get_approx_octmap_density(len(cluster_pc))
        # diagnal min dist to neighbor along 2nd axis
        diag_nn_2d = sqrt(min_dist * min_dist * 2)
        # diagnal min dist to neighbor along 3rd axis
        diag_nn_3d = sqrt(diag_nn_2d * diag_nn_2d + min_dist * min_dist)
        if USE_2D_CL_RADIUS:
          CLUSTER_CL_RADIUS = diag_nn_2d
        else:
          # ARD: changed to 2
          # CLUSTER_CL_RADIUS = 2*diag_nn_3d 
          CLUSTER_CL_RADIUS = diag_nn_3d 
        print("avg distance between octopts:", min_dist, (0.11 * INCH))

        print("min_dist/CLUSTER_CL_RADIUS:", min_dist, CLUSTER_CL_RADIUS)
        # start at top
        c_id = None
        # print("len cluster_pc", len(cluster_pc))

        ##################
        # IDENTIFY INDIV CLUSTERS AND ADD BACK CLUSTER POINTS
        ##################
        octomap_todo = []
        if not FILTER_BASE:
          min_base_z = BIGNUM
          for ob_pt in octobase:
            min_base_z = min(min_base_z, ob_pt[z])
            # ard: use sector computation instead?
        while True:
          if len(octomap_todo) == 0:
            if len(cluster_pc_todo) == 0:
              break
            # create new cluster c_id
            clusters.append([])
            if c_id == None:
              c_id = 0
            else:
              c_id += 1
            if c_id % 50 == 0:
              print("cluster ", c_id)
            # append a new empty list, associated with curr c_id
            # pnt_id = cluster_pc_todo[0]
            pnt_id = cluster_pc_todo.pop(0)
            pnt = cluster_pc[pnt_id]
            # print("pnt1: ", pnt)
            octomap_todo = []
          else:
            # process nested point in exisiting cluster c_id
            pnt_id = octomap_todo[0]
            del octomap_todo[0]  
            pnt = octomap_no_rgb[pnt_id]
            if len(clusters[c_id]) % 200 == 0:
              print("octo_todo", len(octomap_todo), len(cluster_pc_todo), len(clusters[c_id]),pnt_id)
            # print("pnt2: ", pnt)

          # returns a list of the indices of the neighbors of pnt, incl pnt
          neighbors = kdtree.query_ball_point(pnt, r=CLUSTER_CL_RADIUS)
          # all neighbors are in same cluster
          for n in neighbors:
            # n is cluster_pc offset
            # n_pnt = cluster_pc_rgb[n]
            if processed_octomap_pt[n]:
              continue
            n_pnt = octomap[n]
            if pnt[z] <= n_pnt[z]:
              # approx top-down search
              part_of_base = False
              
              if FILTER_BASE:
                z_val = self.base_z[get_sector(n_pnt[0], n_pnt[1])]
                if n_pnt[z] >= z_val:
                  part_of_base = True
                if pt_in_lst(n_pnt, self.w_poss_base_pts):
                  part_of_base = True
              elif n_pnt[z] >= min_base_z:
                for ob_pnt in octobase:
                  if pt_lst_eq(ob_pnt, npnt):
                    part_of_base = True
                    break
          
              if part_of_base:
                # don't process any more of these neighbors
                # print("part of base:",n)
                processed_octomap_pt[n] = True
                # ARD: Allow other neighbors to be processed
                # break
              already_in_clusters = False
              for c_pnt in clusters[c_id]:
                if pt_lst_eq(c_pnt, n_pnt):
                  already_in_clusters = True
                  # print("already in cluster:",n)
                  break
              clusters[c_id].append(n_pnt)
              cluster_pc_todo_idx = None
              for c_idx, c in enumerate(cluster_pc_todo):
                p = cluster_pc[c]
                if pt_lst_eq(p, n_pnt):
                  cluster_pc_todo_idx = c_idx
                  del cluster_pc_todo[c_idx]  
                  break
                elif p[z] > n_pnt[z]:
                  break
              if n not in octomap_todo and cluster_pc_todo_idx == None and not processed_octomap_pt[n]:
                # del cluster_pc_todo[cluster_pc_todo_idx]  
                octomap_todo.append(n)  # to find neighbors of neighbor
              # cluster_pc_todo is a list of cluster_pc offset
              processed_octomap_pt[n] = True
          if len(cluster_pc_todo) == 0 and len(octomap_todo) == 0:
            break
        print("# clusters", len(clusters))
        octoclusters = []
        print("octocluster len:"),
        for c_id in range(len(clusters)):
          # print(c_id, " cluster len:", len(clusters[c_id]))
          if len(clusters[c_id]) >= CLUSTER_MIN_SZ:
            octoclusters.append(clusters[c_id])
            print(len(octoclusters), len(clusters[c_id])),
            # print(len(octoclusters)," octocluster len:", len(clusters[c_id]))
        print("")
        print("# octoclusters", len(octoclusters))
        return octoclusters


    def analyze_pc(self, octomap, octomap_header, min_sample_sz = CLUSTER_MIN_SZ):
      x,y,z = 0,1,2
      self.pc_header = octomap_header
      print("0 analyze_pc len: ", len(octomap))
      ################
      # Initial plane segmentation based on a random subset of the octomap
      # that produces results in less than a minute. 
      # The full set of clusters can be seen by removing the base.
      print("b4 len, len, recomp",  len(self.octobase), len(self.possible_octobase))
      octomap_pc = self.random_pc_subset(np.array(octomap), PC_DENSITY)
      approx_octoclusters_pc, self.octobase = self.segment_octocluster_pc(octomap_pc)
      print("len octobase, len approx_octoclust, recomp",  len(self.octobase), len(approx_octoclusters_pc))
      if len(approx_octoclusters_pc) == 0:
        # no clusters found
        return False
      # ARD HACK
      self.publish_octo_pc(approx_octoclusters_pc, self.octobase, self.pc_header)
      # rospy.sleep(5)
  
      ################
      # Quick additional filtering of the base that leaves clean objects.
      # If there are flat objects that consume a large portion of the base,
      # then turn off FILTER BASE or reduce SECTOR_SIZE.
      if FILTER_BASE:
        print("pre elim base:", len(approx_octoclusters_pc), len(self.octobase))
        self.base_z = compute_z_sectors_from_base(self.octobase)
        delcnt = 0
        for p_id, p in enumerate(approx_octoclusters_pc):
            z_val = self.base_z[get_sector(p[x], p[y])]
            if z_val < p[z]:
              p_array = np.reshape(p, (1,4))
              approx_octoclusters_pc = np.delete(approx_octoclusters_pc,(p_id-delcnt),axis=0)
              self.octobase = np.concatenate((self.octobase,p_array), axis=0)
              delcnt += 1
        print("len approx_octoclusters_pc3:", len(approx_octoclusters_pc), len(self.octobase))
        # print("post elim base:", len(approx_octoclusters_pc), len(self.octobase))
        self.publish_octo_pc(approx_octoclusters_pc, self.octobase, self.pc_header)
        # rospy.sleep(5)

      #################
      # Run additional SEGMENT after filter base_z to further clean up bases
      for filter_num in range(NUM_FILTER_SEGMENT-1):
          save_approx_octoclusters_pc = approx_octoclusters_pc
          save_octobase = self.octobase
          print(filter_num, "segmentation pass:", len(approx_octoclusters_pc), len(self.octobase))
          approx_octoclusters_pc, self.octobase = self.segment_octocluster_pc(approx_octoclusters_pc, self.octobase)
          print("len approx_octoclusters_pc1:", len(approx_octoclusters_pc))
          if len(approx_octoclusters_pc) == 0:
            # no clusters found, restore
            approx_octoclusters_pc = save_approx_octoclusters_pc
            self.octobase = save_octobase
          # rospy.sleep(5)

      #################
      # Identify individual objects from the full set of clusters
      # and add back the full detail of these clusters from the octomap.
      # Store these "pc" clusters and compare them to the previous "world" 
      print("1 analyze_pc len: ", len(octomap))
      self.octoclusters = self.analyze_octoclusters(approx_octoclusters_pc, self.octobase, octomap)
      self.publish_octo_pc(self.octoclusters, self.octobase, self.pc_header, analyzed = True)
      counter = []
      running_sum = []
      for n1 in range(len(self.octoclusters)):
        # Add a new cluster for the pc. 
        # Later, combine clusters with existing cluster?
        self.create_cluster(id = n1, shape = [])
        counter.append(0)
        running_sum.append(np.array([0.0, 0.0, 0.0, 0.0]))

      octoclust = list(self.octoclusters)
      for c_id in range(len(octoclust)):
        # for pnt in self.octoclusters[c_id]:
        for pnt in octoclust[c_id]:
          counter[c_id] += 1
          running_sum[c_id] += pnt
          self.clusters[c_id].cluster['shape'].append(pnt)

      #################
      # compute bounding box.
      # keypoint analysis didn't help much, so currently removed.
      for c_id, c in enumerate(self.clusters):
        # print(c_id," cluster shape:", self.clusters[c_id].cluster['shape'])
        c.compute_bounding_box()
        center = running_sum[c_id] / counter[c_id]
        c.cluster['center'] = center
        # print("center for clust", c , " is ", self.clusters[c].cluster['center'])
        # normalize shape
        c.normalize()
        # print("cluster ",c_id," len", len(c.cluster['shape']))
        # print("cluster", c_id, " obb min,max,rot is ", c.cluster['obb'].min, c.cluster['obb'].max, c.cluster['obb'].rotation)
        # print("cluster", c_id, " centroid is ", c.cluster['obb'].centroid)
        # print(c_id, " cluster shape:", self.clusters[i].cluster['shape'])
      print("num_pc_clusters:",len(self.clusters))

      # check if clusters have flat segments for stacking. 
      # Compare to base_z to check if cluster is potential part of base.
      for c_id, c in enumerate(self.clusters):
        shape_pc = self.clusters[c_id].cluster['shape']
        non_flat_segment, flat_segment = self.segment_octocluster_pc(shape_pc)
        if len(non_flat_segment) == 0:
            # self.clusters[c_id].cluster['shape_attr']
            max_z = -BIGNUM
            for f in flat_segment:
              if max_z < f[z]:
                max_z = f[z]
                z_val = self.base_z[get_sector(f[x], f[y])]
            if abs(max_z - z_val) < 2.1 * OCTOMAP_RESOLUTION:
              print(c_id,": possible base", abs(max_z - z_val))
              self.possible_octobase.append(c_id)
              # print("poss base clust3: ", len(self.possible_octobase))
            # else:
            #   print(c_id,": not part of base", abs(max_z - z_val))
        if len(flat_segment) != 0:
          self.clusters[c_id].set_attr('shape_attr','flat_pc',[flat_segment, self.possible_octobase])
      self.publish_obb()

      return True
   
    #####################
    # BOTH UTILITIES
    #####################
    def get_base_z(self, x, y):
      return self.base_z[get_sector(x, y)]

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
                           location = None, state = None):
      pass

    # def grasp_in_playground(self):
      # self.cluster['centroid'] = centroid          # current centroid 
      # self.cluster['bounding_box'] = bounding_box  # 3 sets of (x,y,z)
      # pass

    #########################################
    # UTILITIES FOR INTEGRATING LATEST PC INTO WORLD 
    #########################################

    def get_pct_overlap(self, w_obb1, pc_obb2):
        w_pc_pct_overlap, pc_w_pct_overlap = OBB.obb_overlap(w_obb1, pc_obb2)
        # print("pct ovrlap:", pct_overlap)
        return w_pc_pct_overlap, pc_w_pct_overlap

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
        return None, None
      # percent overlap
      w_pc_pct_ovrlp, pc_w_pct_ovrlp = self.get_pct_overlap( self.clusters[w_c_id].cluster['obb'],pc_cluster.cluster['obb'])

      # if max_pct_ovrlp != None and max_pct_ovrlp > 0.90:
      # if w_pc_pct_ovrlp != None:
      #   print("obb overlap: world",w_c_id, pc_cluster.cluster['id'], w_pc_pct_ovrlp, pc_w_pct_ovrlp)
      return w_pc_pct_ovrlp, pc_w_pct_ovrlp 

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
      else:
          pc2 = pc
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
    def in_any_obb(self, point, skip_possible_base = True):
        for c_id, c in enumerate(self.clusters):
          if skip_possible_base and c_id in self.possible_octobase:
            continue
          ret = c.in_bounding_box(point)
          if ret[0] == True:
            return c_id
        # print("grasp not in cluster")
        return None

    def assign_grasps(self, grasp_conf):
      # there should be no movement since analyze_pc
  
      found = False
      g_unfound = []
      g_found = []
      if grasp_conf == None:
        return False
      for g_idx, (g, prob) in enumerate(grasp_conf):
        # if no bounding box, find close cluster centers
        d1 = [BIGNUM, BIGNUM]
        d2 = [BIGNUM, BIGNUM]
        d3 = [BIGNUM, BIGNUM]
        # print("len clusters:", len(self.clusters))
        for c_id, c in enumerate(self.clusters):
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

    def analyze_grasp(self, pc):
      self.state = "ANALYZE_GRASP"
      # find cluster that was grabbed / rotated

    def analyze_lift(self, pc):
      self.state = "ANALYZE_LIFT"

    def analyze_drop(self, pc, droppos):
      self.state = "ANALYZE_DROP"
      self.prev_target_dropoff = droppos

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
      g_c_id = self.in_any_obb([grasp[0], grasp[1], grasp[2]])
      return g_c_id

    def rotate_angle(self):
      return DEG20_IN_RADIANS

    def get_action_mode(self):
      pass

    ###################

    def publish_pc_cluster(self, w_c_id):
      print("publish_pc_cluster:", w_c_id, self.w_best_match)
      if w_c_id == None:
        return
      pc_c_id = self.w_best_match[w_c_id]
      clust_pc = self.clusters[pc_c_id].cluster['shape']
      self.publish_octo_pc(clust_pc, [], self.pc_header)
      self.publish_obb(clust_id = pc_c_id)

    def publish_obb(self, filter_poss_base = True, clust_id = None):
      ns = "OBB"
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
      for c_id, c in enumerate(self.clusters):
        if clust_id != None and clust_id != c_id:
          continue
        if c_id in self.possible_octobase:
          continue
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
            m.ns = ns
            m.id = id
            id += 1
      if id < self.obb_prev_count:
        # for i in range(self.obb_prev_count[ns] - id):
        for i in range(self.obb_prev_count - id):
          marker.points = []
          marker.ns = ns
          marker.id = id + i
          markerArray.markers.append(marker)
      self.obb_prev_count = id
  
      # Publish the Marker
      # self.obb_pub.publish(marker)
      self.obb_pub.publish(markerArray)
