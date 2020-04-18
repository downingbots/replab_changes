#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
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
      self.num_clusters = 0
      self.clusters = []
      self.clusters_history = []
      self.octoclusters = []
      self.octobase = []
      self.pc_octocluster_pub = rospy.Publisher(PC_OCTOCLUSTER_TOPIC, PointCloud2, queue_size=1)
      self.pc_octobase_pub = rospy.Publisher(PC_OCTOBASE_TOPIC, PointCloud2, queue_size=1)
      self.obb_pub = rospy.Publisher(OBB_TOPIC, MarkerArray, queue_size=1)

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
      self.num_clusters = pc_clusters.num_clusters
      self.clusters = pc_clusters.clusters


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

    def copy_curr_pc_to_world_state(self, pc_clusters):
        print("copy over pc clusters")
        # just store the pc_clusters 
        self.clusters_history.append(self.clusters)
        self.clusters = pc_clusters.clusters
        self.save_world()

    # integrates temporary "pc cluster" into persistent "world" cluster
    def integrate_current_pc_into_world(self, pc_clusters):
      if self.world_state == 'UNINITIALIZED' or len(self.clusters) == 0:
        self.copy_curr_pc_to_world_state(pc_clusters)
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
      obb_max_matching_clusters = [None for i, w_c in enumerate(self.clusters)]
      obb_min_matching_clusters = [None for i, w_c in enumerate(self.clusters)]
      obb_max_max_pct_ovlp = [0 for i, w_c in enumerate(self.clusters)]
      obb_max_min_pct_ovlp = [0 for i, w_c in enumerate(self.clusters)]
      obb_ovlp_cnt = [0 for i, w_c in enumerate(self.clusters)]

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
            max_obb_pct_ovlp, min_obb_pct_ovlp = self.compare_bounding_box(w_c_id, pc_c)

            if obb_max_max_pct_ovlp[w_c_id] < max_obb_pct_ovlp:
              obb_max_matching_clusters[w_c_id] = i
              obb_max_max_pct_ovlp[w_c_id] = max_obb_pct_ovlp

            if obb_max_min_pct_ovlp[w_c_id] < min_obb_pct_ovlp:
              obb_min_matching_clusters[w_c_id] = i
              obb_max_min_pct_ovlp[w_c_id] = min_obb_pct_ovlp

            # indicator that we should combine clusters
            if max_obb_pct_ovlp > .70 or min_obb_pct_ovlp > .70:
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
        obb_max_match   = obb_max_matching_clusters[w_c_id]
        obb_min_match   = obb_min_matching_clusters[w_c_id]
        obb_max_ovrlp   = obb_max_max_pct_ovlp[w_c_id]
        obb_min_ovrlp   = obb_max_min_pct_ovlp[w_c_id]
        obb_ovrlp_num   = obb_ovlp_cnt[w_c_id]

      if len(pc_unmatched) > 0:
        print("len unmatched PC clusters: ", len(pc_unmatched))
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
        print("unmatched PC clusters len: ", len(pc_unmatched))
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
      self.copy_curr_pc_to_world_state(pc_clusters)

################ END integration

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

    def publish_octo_pc(self, octoclusters, octobase, header):
      if not DISPLAY_PC_CLUSTERS:
        return
      for i in range(2):
        # cluster_pc = []
        if i == 0:
          print("octoclusters")
          cluster_pc = octoclusters
          # for c in octoclusters:
            # for p in c:
              # cluster_pc.append(p)
          # print("len octoclusters", len(octoclusters))
        else:
          print("octobase")
          cluster_pc = octobase
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
    # https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/geometry/models/plane.py
    # https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/ransac/fitters.py
    #####################################################
    def find_octoclusters_octobase_pcs(self, pc):
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
      print("mean ", self.point)
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
      max_dist = 1.1 * get_approx_octmap_density(len(pc))
      print("max_dist",max_dist, n_inliers_to_stop)
      print_once = True
      for i in range(max_iterations):
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
              print("normal: ",normal)
            self.normal = [1,1,1]
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
      print("len plane: ", len(best_inliers))
      print("len pc   : ", len(pc))             # same len as above
      octobaselst = []
      octoclusterslst = []
      for i in range(len(best_inliers)):
        if best_inliers[i]:
          octobaselst.append(pc[i])
        else:
          octoclusterslst.append(pc[i])
      octobase = np.array(octobaselst)[:, :4]
      print("octobase shape", octobase.shape)
      print("octoclusters len", len(octoclusterslst))
      if len(octoclusterslst) == 0:
        octoclusters = octobase
        print("set octoclusters to octobase")
      else:
        octoclusters = np.array(octoclusterslst)[:, :4]
        print("octoclusters shape", octoclusters.shape)
      return octoclusters, octobase

    def analyze_octoclusters(self, octoclusters_pc):
      # find the individual objects in the combined octocluster pcs
        def take_z_axis(elem):
          return elem[2]

        z = 2
        # pc is cluster only; tray has been filtered out
        # Start with the highest points, and compute neighbors in cluster.
        # Sorting used by cluster analysis to compute distances
        cluster_pc_rgb = sorted(octoclusters_pc, key=take_z_axis)
        cluster_pc = np.array(cluster_pc_rgb)[:, :3]
        cluster_pc_todo = [i for i in range(len(cluster_pc))]
        USE_2D_CL_RADIUS = False
        if USE_2D_CL_RADIUS:
          cluster_pc_2d = np.array(cluster_pc_rgb)[:, :2]

        ##################
        # CLUSTER ANALYSIS
        ##################
        sys.setrecursionlimit(20000)
        kdtree       = spatial.KDTree(cluster_pc)
        if USE_2D_CL_RADIUS:
          kdtree_2d    = spatial.KDTree(cluster_pc_2d)
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
          CLUSTER_CL_RADIUS = diag_nn_3d 
        min_dist = 1.1 * get_approx_octmap_density(len(cluster_pc))
        print("avg distance between octopts:", min_dist, (0.11 * INCH))

        print("min_dist/CLUSTER_CL_RADIUS:", min_dist, CLUSTER_CL_RADIUS)
        # start at top
        c_id = None
        # print("len cluster_pc", len(cluster_pc))

        cluster_member_todo = []
        while True:
          if len(cluster_member_todo) == 0:
            if len(cluster_pc_todo) == 0:
              break
            # create new cluster c_id
            clusters.append([])
            if c_id == None:
              c_id = 0
            else:
              c_id += 1
            # append a new empty list, associated with curr c_id
            pnt_id = cluster_pc_todo[0]
            cluster_pc_todo.remove(pnt_id)
            pnt = cluster_pc[pnt_id]
            # print("pnt1: ", pnt)
            cluster_member_todo = []
          else:
            # process nested point in exisiting cluster c_id
            pnt_id = cluster_member_todo[0]
            cluster_member_todo.remove(pnt_id) 
            pnt = cluster_pc[pnt_id]
            # print("pnt2: ", pnt)

          # returns a list of the indices of the neighbors of pnt, incl pnt
          neighbors = kdtree.query_ball_point(pnt, r=CLUSTER_CL_RADIUS)
          # all neighbors are in same cluster
          for n in neighbors:
            # n is cluster_pc offset
            n_pnt = cluster_pc_rgb[n]
            if pnt[z] <= n_pnt[z]:
              # top-down search
              already_in_clusters = False
              for c in clusters[c_id]:
                if c[0] == pnt[0] and c[1] == pnt[1] and c[2] == pnt[2]:
                  already_in_clusters = True
                  break
              if not already_in_clusters:
                clusters[c_id].append(n_pnt)
              if n not in cluster_member_todo and n in cluster_pc_todo:
                cluster_pc_todo.remove(n)  
                cluster_member_todo.append(n)  # to find neighbors of neighbor
              # cluster_pc_todo is a list of cluster_pc offset
          if len(cluster_pc_todo) == 0 and len(cluster_member_todo) == 0:
            break
        print("# clusters", len(clusters))
        octoclusters = []
        for c_id in range(len(clusters)):
          print(c_id, " cluster len:", len(clusters[c_id]))
          if len(clusters[c_id]) >= CLUSTER_MIN_SZ:
            octoclusters.append(clusters[c_id])
        print("# octoclusters", len(octoclusters))
        return octoclusters

    def analyze_pc(self, octomap, octomap_header, min_sample_sz = CLUSTER_MIN_SZ):
      self.pc_header = octomap_header
      octoclusters_pc, self.octobase = self.find_octoclusters_octobase_pcs(octomap)
      self.publish_octo_pc(octoclusters_pc, self.octobase, self.pc_header)
      print("analyze_pc len: ", len(octomap))
      self.octoclusters = self.analyze_octoclusters(octoclusters_pc)
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
                  # pnt = octoclust[c_id]
                  counter[c_id] += 1
                  running_sum[c_id] += pnt
                  # print(c_id, "shape append", pnt)
                  self.clusters[c_id].cluster['shape'].append(pnt)
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
      self.publish_obb()
      return True
   

    #####################################################
 
    # analyze latest point cloud into a set of clusters
    def analyze_pc_old(self, pc, min_sample_sz = CLUSTER_MIN_SZ):
      print("analyze_pc len: ", len(pc))
      if len(pc) < DBSCAN_MIN_SAMPLES:
        return False
      print("analyze_pc pc[0]: ", pc[0])
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
      for n1 in range(n1_clusters):
        # Add a new cluster for the pc. 
        # Later, combine clusters with existing cluster?
        self.create_cluster(id = n1, shape = [])
        counter.append(0)
        running_sum.append(np.array([0.0, 0.0, 0.0, 0.0]))
        # running_sum.append(np.array([0.0, 0.0, 0.0]))

      print("len labels vs len pc[i]:", len(db1.labels_), len(pc))
      for c_id in set(db1.labels_):
        if c_id != -1:
          for i, label in enumerate(db1.labels_):
              if db1.labels_[i] == c_id:
                  # print("label", c_id, i, pc[i])
                  counter[c_id] += 1
                  running_sum[c_id] += pc[i]
                  # print(c_id, "shape append", pc[i])
      for c_id, c in enumerate(self.clusters):
        c.compute_bounding_box()
        center = running_sum[c_id] / counter[c_id]
        c.cluster['center'] = center
        # print("center for clust", c , " is ", self.clusters[c].cluster['center'])
        # normalize shape
        c.normalize()
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
                           location = None, state = None):
      pass

    # def grasp_in_playground(self):
      # self.cluster['centroid'] = centroid          # current centroid 
      # self.cluster['bounding_box'] = bounding_box  # 3 sets of (x,y,z)
      # pass

    #########################################
    # UTILITIES FOR INTEGRATING LATEST PC INTO WORLD 
    #########################################

    def get_pct_overlap(self, obb1, obb2):
        max_pct_overlap, min_pct_overlap = OBB.obb_overlap(obb1, obb2)
        max_pct_overlap, min_pct_overlap = OBB.obb_overlap(obb1, obb2)
        # print("pct ovrlap:", pct_overlap)
        return max_pct_overlap, min_pct_overlap

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
      max_pct_ovrlp, min_pct_ovrlp = self.get_pct_overlap( self.clusters[w_c_id].cluster['obb'],pc_cluster.cluster['obb'])
      if max_pct_ovrlp != None and max_pct_ovrlp > 0.90:
        print("obb overlap: world",w_c_id, pc_cluster.cluster['id'], min_pct_ovrlp, max_pct_ovrlp)
      return max_pct_ovrlp, min_pct_ovrlp 

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
    # World
    #################
    def in_any_obb(self, point):
        for c_id, c in enumerate(self.clusters):
          ret = c.in_bounding_box(point)
          if ret[0] == True:
            print("grasp in cluster",c_id)
            return True
        # print("grasp not in cluster")
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
