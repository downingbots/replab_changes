#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
from sensor_msgs.msg import (Image)

from utils_grasp import *
from config_grasp import *
import utils_grasp
import cv2 as cv

class ClusterState:

    def __init__(self, pc, KP):
      self.cluster_centers = []
      self.cluster_shapes = []
      self.cluster_KPs = []
      self.KP = KP
      self.analyze_pc(pc, KP)
      self.state = "NEW_GRASP"
      self.state_change = "NEW"
      if DISPLAY_PC_CLUSTERS:
        self.pc_cluster_pub = rospy.Publisher(PC_CLUSTER_TOPIC, PointCloud2, queue_size=1)

    def publish(self, img):
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


    def analyze_grasp(self, pc):
      self.state = "ANALYZE_GRASP"
      # find cluster that was grabbed / rotated

    def analyze_lift(self, pc):
      self.state = "ANALYZE_LIFT"

    def analyze_drop(self, pc):
      self.state = "ANALYZE_DROP"

    def save_history(self):

    def analyze_next_pc(self, pc, KP):
        
      self.cluster_centers = []
      self.cluster_shapes = []
      self.cluster_KPs = []
      self.state = "NEW_GRASP"

    def analyze_pc(self, pc, KP):
      db1 = DBSCAN(eps=.001, min_samples=3,
                   n_jobs=-1).fit(pc1)
      # Number of clusters in labels, ignoring noise if present.
      n1_clusters = len(set(db1.labels_)) - (1 if -1 in db1.labels_ else 0)
      print("DBSCAN: # ", n1_clusters)
      self.cluster_centers = []
      self.cluster_shapes = []
      self.cluster_KPs = []
      self.num_clusters = 0
      kp = KP.get_kp() 

      for cluster in set(db1.labels):
        if cluster != -1:
          running_sum = np.array([0.0, 0.0, 0.0])
          counter = 0
          cluster_kp = []
          cluster_shape = []
          self.num_clusters += 1 
          for i in range(pc.shape[0]):
              if db1.labels[i] == cluster:
                  running_sum += pc[i]
                  counter += 1
                  cluster_pc.append(pc[i])
                  if [pc[i][0], pc[i][1]] in kp:
                    cluster_kp.append(pc[i])
                    print("kp found in cluster")
          center = running_sum / counter
          # ClusterState: [id, center, KPs, pc]
          self.cluster_id.append(cluster)
          self.cluster_centers.append(center)
          self.cluster_KPs.append(cluster_kp)
          self.cluster_PCs.append(cluster_pc)


    def cluster_history():

    def compare_clusters(CL2):

    def combine_clusters(CL2):
      import numpy as np
      import time
      import icp

      T, distances, iterations = icp.icp(cl, CL2, tolerance=0.000001)



