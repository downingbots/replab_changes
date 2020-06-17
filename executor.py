#!/usr/bin/env python

import numpy as np
from matplotlib.patches import Circle
import h5py
import math

import rospy
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import (
    Image,
    PointCloud2,
    CameraInfo,
    JointState
)
from std_msgs.msg import (
    UInt16,
    Int64,
    Float32,
    Header
)
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError

from moveit_commander import PlanningSceneInterface
from moveit_commander.exception import MoveItCommanderException

from random import seed
from random import random

from itertools import product
import time
import traceback
import argparse

from sklearn.neighbors import KDTree
from scipy.ndimage import rotate

from replab_core.controller import WidowX
from replab_core.config import *
from replab_core.utils import *
# from replab_grasping.utils_grasp import *
# from replab_grasping.config_grasp import *
from utils_grasp import *
from config_grasp import *
from action_history import *

class Executor:

    def init_manual_grasp_calibration_history(self):
        grasp_history = [
                       [0, [-0.00082, -0.01508, 0.4751, -0.21755]],
                       [1, [-0.10245, 0.00702, 0.4826, -0.06402]],
                       [2, [-0.15773, 0.0895, 0.4826, -0.17537]],
                       [3, [0.02615, -0.09245, 0.4826, -0.16235]],
                       [4, [-0.06592, 0.12449, 0.4801, -0.14054]],
                       [5, [0.05676, -0.07912, 0.4826, 0.03237]],
                       [6, [0.09989, -0.1333, 0.39625, -0.83611]],
                       [7, [0.05079, -0.07826, 0.4876, 0.25279]],
                       [8, [-0.05168, 0.1272, 0.4826, -0.18158]],
                       [9, [0.10369, -0.13674, 0.39625, -0.54639]],
                       [10, [-0.06796, 0.10486, 0.4801, -0.0684]],
                       [11, [0.09286, -0.14802, 0.4825, 0.0414]],
                       [12, [0.10369, -0.13674, 0.39625, -0.54639]],
                       [13, [-0.06796, 0.10486, 0.4801, -0.0684]],
                       [14, [0.09286, -0.14802, 0.4825, 0.0414]],
                       [15, [-0.06795, 0.10472, 0.4801, -0.09991]],
                       [16, [0.09286, -0.14802, 0.4825, 0.0414]],
                       [17, [0.01294, -0.08313, 0.4826, 0.00401]],
                       [18, [0.09237, -0.1478, 0.48125, 0.18493]],
                       [19, [0.04015, -0.13574, 0.4826, -0.00422]],
                       [20, [-0.08985, -0.10168, 0.4876, -0.0268]],
                       [21, [0.06368, 0.13282, 0.4726, 0.56394]],
                       [22, [0.03994, 0.08068, 0.4776, 0.11742]],
                       [23, [-0.05582, 0.02392, 0.4801, -0.05587]],
                       [24, [-0.01814, -0.06826, 0.4801, -0.75024]],
                       [25, [-0.12386, -0.04147, 0.47375, -0.08003]],
                       [26, [-0.01004, 0.07502, 0.4801, 0.04323]],
                       [27, [-0.02633, -0.06617, 0.47625, 0.59637]],
                       [28, [0.08493, 0.07904, 0.46875, 0.06513]],
                       [29, [0.06057, 0.0392, 0.47875, -0.52877]],
                       [30, [-0.15401, 0.0859, 0.4801, -0.14314]],
                       [31, [-0.00461, -0.13922, 0.4826, -0.02725]],
                       [32, [-0.01709, -0.05971, 0.4801, -0.01963]],
                       [33, [-0.1552, 0.13042, 0.4876, -0.31355]],
                       [34, [-0.04139, -0.1313, 0.4851, 0.2989]],
                       [35, [-0.04166, -0.13183, 0.4851, 0.32844]],
                       [36, [-0.04163, -0.13174, 0.4851, 0.3428]],
                       [37, [0.05679, -0.1403, 0.4826, 0.11431]],
                       [38, [-0.04154, -0.13146, 0.4851, 0.25263]],
                       [39, [0.16071, -0.01325, 0.4801, -0.78173]],
                       [40, [-0.03693, 0.14142, 0.47375, 0.18383]],
                       [41, [0.16071, -0.01325, 0.4801, -0.78173]],
                       [42, [0.16979, -0.02375, 0.4726, -0.79948]],
                       [43, [-0.10207, -0.13158, 0.4901, -0.30687]],
                       [44, [-0.16716, 0.04625, 0.44375, -0.66597]]
                       ]


        grasp_delta_history = [
 [0, [-0.015000000000000001, -0.0075, 0, -0.47123889803846897, 'EXECUTE_GRASP']],
 [1, [-0.02, 0.029999999999999995, 0, 0, 'EXECUTE_GRASP']],
 [2, [-0.024999999999999998, 0.04750000000000001, 0, 0, 'EXECUTE_GRASP']],
 [3, [0, 0, 0, 0, 'EXECUTE_GRASP']],
 [4, [0, 0.029999999999999995, 0, 0.15707963267948966, 'EXECUTE_GRASP']],
 [5, [-0.027499999999999997, -0.01, 0, 0, 'EXECUTE_GRASP']],
 [6, [-0.04750000000000001, -0.06000000000000002, 0, 0, 'EXECUTE_GRASP']],
 [7, [0.015000000000000001, 0.0175, 0, 0, 'EXECUTE_GRASP']],
 [8, [0, 0.027499999999999997, 0, 0, 'EXECUTE_GRASP']],
 [9, [-0.05750000000000001, -0.06000000000000001, 0, -0.47123889803846897, 'EXECUTE_GRASP']],
 [10, [0, 0.029999999999999995, 0, 0.15707963267948966, 'EXECUTE_GRASP']],
 [11, [-0.045000000000000005, 0, 0, -1.0995574287564276, 'EXECUTE_GRASP']],
 [12, [-0.05750000000000001, -0.06000000000000001, 0, -0.47123889803846897, 'EXECUTE_GRASP']],
 [13, [0, 0.029999999999999995, 0, 0.15707963267948966, 'EXECUTE_GRASP']],
 [14, [-0.045000000000000005, 0, 0, -1.0995574287564276, 'EXECUTE_GRASP']],
 [15, [-0.02, -0.0075, 0, -0.3141592653589793, 'EXECUTE_GRASP']],
 [16, [-0.032499999999999994, -0.005, 0, -1.0995574287564276, 'EXECUTE_GRASP']],
 [17, [0.01, -0.015000000000000001, 0, 0, 'EXECUTE_GRASP']],
 [18, [0, -0.024999999999999998, 0, -0.6283185307179586, 'EXECUTE_GRASP']],
 [19, [-0.0375, -0.0225, 0, 0, 'EXECUTE_GRASP']],
 [20, [-0.05250000000000001, 0.015000000000000001, 0, 0, 'EXECUTE_GRASP']],
 [21, [0.02, 0.0125, 0, 0, 'EXECUTE_GRASP']],
 [22, [-0.0075, 0.015000000000000001, 0, 0, 'EXECUTE_GRASP']],
 [23, [-0.029999999999999995, 0.0175, 0, 0.47123889803846897, 'EXECUTE_GRASP']],
 [24, [-0.024999999999999998, 0.0125, 0, -0.7853981633974483, 'EXECUTE_GRASP']],
 [25, [-0.0175, -0.032499999999999994, 0, 1.413716694115407, 'EXECUTE_GRASP']],
 [26, [-0.0075, 0.0175, 0, -2.0420352248333655, 'EXECUTE_GRASP']],
 [27, [-0.04750000000000001, 0, 0, 0, 'EXECUTE_GRASP']],
 [28, [0.01, 0, 0, 1.413716694115407, 'EXECUTE_GRASP']],
 [29, [0.01, 0.0025, 0, 1.8849555921538759, 'EXECUTE_GRASP']],
 [30, [-0.0125, 0.0375, 0, 0, 'EXECUTE_GRASP']],
 [31, [-0.045, -0.015, 0, 0, 'EXECUTE_GRASP']],
 [32, [-0.0325, 0, 0, 0.3141592653589793, 'EXECUTE_GRASP']],
 [33, [-0.0275, 0.0475, 0, 0.9424777960769379, 'EXECUTE_GRASP']],
 [34, [-0.04750000000000001, 0, 0, 0.6283185307179586, 'EXECUTE_GRASP']],
 [35, [0, 0, 0, 0, 'EXECUTE_GRASP']],
 [36, [-0.0175, 0, 0, 0.6283185307179586, 'EXECUTE_GRASP']],
 [37, [-0.03, -0.03, 0, 0, 'EXECUTE_GRASP']],
 [38, [-0.0075, -0.0075, 0, 0, 'EXECUTE_GRASP']],
 [39, [0.0025, 0, 0, 0.3141592653589793, 'EXECUTE_GRASP']],
 [40, [0, 0.02, 0, -2.0420352248333655, 'EXECUTE_GRASP']],
 [41, [0, -0.015, 0, 0, 'EXECUTE_GRASP']],
 [42, [0, -0.0075, 0, 0, 'EXECUTE_GRASP']],
 [43, [-0.0575, 0, 0, 0, 'EXECUTE_GRASP']],
 [44, [-0.025, -0.0245, 0, -0.3141592653589793, 'EXECUTE_GRASP']]
 ]
        for i, g_h in enumerate(grasp_history):
          [num, g] = grasp_history[i]
          [num, g_d] = grasp_delta_history[i]
          self.manual_grasp_calibration_history.append([g, g_d, True])
        self.manual_grasp_calibration_print_history()


    def __init__(self, scan=False, datapath='', save=False):
        # WidowX controller interface
        self.widowx = WidowX()
        self.history = ActionHistory()
        self.transform = lambda x: x
        # For ROS/cv2 conversion
        self.bridge = CvBridge()
        self.pc_ready = True
        self.pc_count = 0
        self.octomap_ready = False
        self.octomap_ready2 = False
        self.manual_grasp_calibration_history = []
        self.init_manual_grasp_calibration_history()
        self.calibration_matrix2 = None 
        self.residuals2 = None
        self.sector_grasp_delta = []
        self.calibration_mode = False
        for i in range(SECTOR_SIZE * SECTOR_SIZE):
          self.sector_grasp_delta.append([0, 0, 0, 0, 0])

        # Register subscribers
        self.pc_subscriber = rospy.Subscriber(
            POINTCLOUD_TOPIC, PointCloud2, self.update_pc)
        self.octomap_pc_subscriber = rospy.Subscriber(
            OCTOMAP_TOPIC, PointCloud2, self.update_octomap)

        # Register cluster pub
        # pub transformed pc to octomap server
        self.pc_rgb_pub1 = rospy.Publisher(PC_RGB_TOPIC, PointCloud2, queue_size=1)
        self.pc_octomap_pub2 = rospy.Publisher(PC_OCTOMAP_TOPIC, PointCloud2, queue_size=5)

        if DISPLAY_PC_GRASPS:
          self.pc_grasps_pub3 = rospy.Publisher(PC_GRASPS_TOPIC, PointCloud2, queue_size=1)

        self.drop_eeb = [100,100]
        for eeb in END_EFFECTOR_BOUNDS:
          self.drop_eeb[0] = min(self.drop_eeb[0], abs(eeb[0]))
          self.drop_eeb[1] = min(self.drop_eeb[1], abs(eeb[1]))
        self.datapath = datapath
        self.save = save

        # seed random number generator (for now be predictable)
        seed(1)

        # Tracking misses for calling reset routine
        self.running_misses = 0

        self.evaluation_data = []

        # Store latest RGB-D

        self.pc = []                   # unmaterialized pc
        self.transformed_pc_rgb = []   
        self.base_z = None
        self.sample = {}
        self.pc_header = None
        self.octomap_pc = []
        # p = [0.16999318981759565, 0.14906749041266876, 0.47066674113892565, np.uint32(9465628)]
        # for x in range(136):
        #   for y in range(120):
        #     for z in range(9):
        #       self.octomap_pc.append(p)
        # self.octomap_count = 0
        self.octomap_header = None

        self.camera_info = CameraInfo()
        if scan:
            self.base_pc = np.zeros((1, 3)) + 5000.0  # adds dummy point
        else:
            self.base_pc = np.load(PC_BASE_PATH)
        self.cm = CALIBRATION_MATRIX
        self.inv_cm = np.linalg.inv(self.cm)
        self.kdtree = KDTree(self.base_pc)
        self.recent_grasps = []

        rospy.sleep(2)

        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(self.camera_info)

    def update_octomap(self, data):
        if not self.octomap_ready:
          self.octomap_ready2 = False
          return
        self.octomap_header = data.header
        self.octomap = []
        ##  for pt in pc2.read_points(data, skip_nans=True):
        #    self.octomap.append([pt[0], pt[1], pt[2], pt[3]])
        self.octomap = list(pc2.read_points(data, skip_nans=True))
        self.octomap_ready2 = True
        
        # self.octomap = pc2.read_points(data, skip_nans=True)
        # width = 136  (.34 meters @ .0025)
        # height = 120 (.3 meters @ .0025)
        # x = 136
        # y = 120 
        # z = 9
        # width = 4
        # height = 136 * 120 * 9
        # self.octomap = pc2.read_points(data, skip_nans=True,  uvs=[[4, height]])
        # self.octomap_count += 1
        # print("octo_cnt: ", self.octomap_count, len(self.octomap))

    # point cloud topic 
    def update_pc(self, data):
        if not self.pc_ready:
          # busy materializing from previous pointcloud.
          # skip and process next point cloud instead
          # print("skip u_pc;"),
          return
        self.pc_header = data.header
        # print("pc_header: ", self.pc_header)
        self.pc = pc2.read_points(data, skip_nans=True)
        # self.pc = list(pc2.read_points(data, skip_nans=False))
        self.pc_count = self.pc_count + 1
        if self.pc_count == 10:
          # self.pub_cluster()
          self.pc_count = 0

    def save_cinfo(self, data):
        self.camera_info = data

    def get_pose(self):
        pose = self.widowx.get_current_pose().pose
        pose_list = [pose.position.x,
                     pose.position.y,
                     pose.position.z,
                     pose.orientation.w,
                     pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z]
        return pose_list

    # not used by pick_push
    def get_rgbd(self):
        old_depth = self.depth.astype(np.float)
        depth = rectify_depth(old_depth)
        depth = np.reshape(depth, (480, 640, 1))
        # compute keypoints only before use / call get_rgbd() before get_pc()
        # self.keypoints = self.find_features_2d(self.rgb)
        return np.concatenate([self.rgb, depth], axis=2)


    def get_octomap_pc(self):
        ####################################
        # ARD: Experiment
        # pc = self.get_pc()
        # np.random.shuffle(pc)
        # if len(pc) > 12000:
        #   pc = pc[:12000]
        # return pc, self.pc_header
        ####################################
        # while self.pc_count != self.octomap_count:
        #   print(" self.pc_count:", self.pc_count," octomap_count:", self.octomap_count)
        #   rospy.sleep(1)
        for warmup in range(OCTOMAP_WARMUP):
          print("warmup:", warmup)
          if warmup > 0:
            # publish /pc_rgb
            print("get_pc")
            # Didn't speed things up:
            # for prewarmup in range(OCTOMAP_WARMUP):
            #   self.get_pc()  
          self.octomap_pc = []
          retry_cnt = 0
          while len(self.octomap_pc) == 0:
            self.octomap_ready = True
            if self.octomap_ready2:
              self.octomap_pc = self.octomap
              # self.octomap_pc = list(self.octomap)
              # self.octomap_pc = [[p[0],p[1],p[2],np.uint32(p[3])] for p in self.octomap]
              self.octomap_ready = False
              self.octomap_ready2 = False
              print((OCTOMAP_WARMUP - warmup), " len octomap_pc:",len(self.octomap_pc))
            # if len(self.octomap_pc) > 0:
            #   print("octomap[0]:",self.octomap_pc[0])
            if len(self.octomap_pc) == 0:
              # if retry_cnt >= 3:
              if retry_cnt >= 0:
                print("get_pc")
                retry_cnt = 0
                self.get_pc()  
              else:
                retry_cnt += 1
              # print("sleep2")
              # rospy.sleep(2)
          # octomap_pc = np.array(self.octomap_pc)[:, :4]
          # print(" octomap_pc[0]:",self.octomap_pc[0])
        # octomap returns average color, so no need to use color from closest point in transformed pc.
        print("done warmup")
        self.octomap_pc = add_color(self.octomap_pc, self.transformed_pc_rgb)
        # print("2 octomap_pc[0]:",self.octomap_pc[0])
        
        # self.publish_pc(PC_OCTOMAP_TOPIC, self.octomap_pc)
        return self.octomap_pc, self.octomap_header

    def get_pc(self):
        # materialize pc
        self.pc_ready = False
        pc = list(self.pc)
        self.pc_ready = True
        # print("len self.pc ", len(pc))
        if len(pc) == 0:
          print("No point cloud from camera!")
          return None
        rgb_pc = np.array(pc)[:, :4]
        # np.random.shuffle(rgb_pc)
        # print("rgb_pc shape0 ", rgb_pc.shape[0])
        print("rgb_pc shape ", rgb_pc.shape)
        pc = np.array(rgb_pc)[:, :3]

        def transform_pc(srcpc, tf_matrix):
            ones = np.ones((srcpc.shape[0], 1))
            srcpc = np.append(srcpc, ones, axis=1)
            if not self.calibration_mode and self.calibration_matrix2 != None :
              cm2 = np.dot(self.calibration_matrix2, tf_matrix.T)
              out = np.dot(cm2, srcpc.T)
            else:
              out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc = transform_pc(pc, self.cm)
        # if not self.calibration_mode:
        #   cm2  = np.array([[-1.0,  0.0,  0.0, 0.0],
        #                    [ 0.0, -0.95,  0.39, 0.0],
        #                    [ 0.0,  0.47,  0.95, 0.0],
        #                    [ 0.0,  0.0,  0.0, 1.0]])
        #   cm2 = self.cm
        #   pc = transform_pc(pc, cm2)

        # put rgb back in pc
        # rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS)]
        rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS)]
        pc = rgb_pc
        # print("len pc ", len(pc))

        # pruning not required with Octomap. Higher res is better.
        # # do further PC size reduction for kdtree and grip analysis
        # if len(pc) > PC_DENSITY:
        #     pc = pc[:PC_DENSITY]
        # print("pruned pc len:",len(pc))

        pc = np.reshape(pc, (len(pc), 4))
        self.transformed_pc_rgb = pc
        self.publish_pc(PC_RGB_TOPIC, self.transformed_pc_rgb)
        return pc

#############################################33
# Many variations to filter base were tried before settling on the
# current plane segmentation with base_filter approach.
#
# The cluster set is found by eliminating the base.
#
#     def filter_base(self,pc):
#         ######################
#         # Filter out base tray
#         ######################
#         
#         return pc
# 
# 
#         # z "elbow" to filter base
#         return rm_base(pc)
# 
#         plane = segment_cluster(pc)
#         p1 = [p for i,p in enumerate(pc) if not plane[i]]
#         p2 = [p for i,p in enumerate(pc) if plane[i]]
#         print("prefilter pc len:",len(pc), len(plane), len(p1), len(p2))
#         pc2_rgb = p2
#         print("postfilter pc len:",len(pc2_rgb))
#         return pc2_rgb
# 
# 
#         ######################
#         # z "elbow" to filter base
#         return rm_base(pc)
# 
#         ######################
#         # use segmentation to filter base
#         ######################
#         print("prefilter pc len:",len(pc))
#         # ARD: any (base_z - MIN_OBJ_HEIGHT) greater than .485 is suspect...
#         # print(MIN_OBJ_HEIGHT, " sector base_z:", self.base_z)
# 
#         plane = segment_cluster(pc)
#         pc2_rgb = []
#         for i, p in enumerate(pc):
#           if not plane[i]:
#              pc2_rgb.append(p)
#         print("postfilter pc len:",len(pc2_rgb))
#         return pc2_rgb
# 
#         ######################
#         # use averages to filter base
#         ######################
#         self.base_z = compute_z_sectors(pc, self.base_z)
#         pc2_rgb = []
#         for i, p in enumerate(pc):
#              sect = get_sector(p[0],p[1])
#              if (p[2] > self.base_z[sect] - MIN_OBJ_HEIGHT):
#                if i < 10:
#                  print("TOO CLOSE TO GROUND ", p[2], self.base_z[sect], MIN_OBJ_HEIGHT)
#                continue
#              # if (p[2] > 0.485):
#              #   print("suspect sector: ", sect, self.base_z[sect], p)
#              pc2_rgb.append(p)
#         print("postfilter pc len:",len(pc2_rgb))
#         ######################
#         return pc2_rgb

    def get_calibrated_grasp(self, grasp):
      new_grasp = grasp
      sect = get_sector(grasp[0], grasp[1])
      for i in range(4):
        new_grasp += sector_grasp_delta[sect][i] / sector_grasp_delta[sect][4]
      return new_grasp

    def calibrate_grasp_compute_sectors(self, prev_grasp_delta = None):
      sector_sz = SECTOR_SIZE  # depends on max object size vs. base sz
      if self.manual_grasp_calibration_history == None or len(self.manual_grasp_calibration_history) == 0:
        return None
      [grasp, grasp_delta, result] = self.manual_grasp_calibration_history[0]
      if prev_grasp_delta != None:
        grasp_grasp_delta = prev_grasp_delta
      else:
        sector_grasp_delta = []
        for i in range(sector_sz * sector_sz):
          sector_grasp_delta.append([0, 0, 0, 0, 0])
      for [grasp, grasp_delta, result] in self.manual_grasp_calibration_history:
        [x, y, z, theta] = grasp
        [delta_x, delta_y, delta_z, delta_theta, new_action] = grasp_delta
        sect = get_sector(x, y)
        for i in range(4):
          sector_grasp_delta[sect][i] += grasp_delta[i] 
      sector_grasp_delta[sect][4] += len(self.manual_grasp_calibration_history)
      print("sector_grasp_delta: ", sector_grasp_delta )
      return sector_grasp_delta

    def manual_grasp_calibration(self, grasp_delta):
        [delta_x, delta_y, delta_z, delta_theta, new_action] = grasp_delta
        resolution = .0025 
        theta_resolution = np.pi / 20
        print("arm xy (wasd), theta (jl), z (ik), execute grasp (space), skip grasp (;), next img (ret):")
        c = getChar() 
        if c == 'd':
          delta_x -= resolution
        elif c == 'a':
          delta_x += resolution
        elif c == 'w':
          delta_y += resolution
        elif c == 'x' or c == 's':
          delta_y -= resolution
        elif c == 'i':
          delta_y += resolution
        elif c == 'k':
          delta_y -= resolution
        elif c == 'j':
          delta_theta -= theta_resolution
        elif c == 'l':
          delta_theta += theta_resolution
        elif c == ';':
          new_action = "SKIP_GRASP"
        elif c == ' ':
          new_action = "EXECUTE_GRASP"
        else:
          new_action = "NEXT_IMAGE"
        return [delta_x, delta_y, delta_z, delta_theta, new_action]

    def manual_grasp_calibration_print_history(self):
        self.robot_coordinates  = []
        self.camera_coordinates = []
        self.sector_grasp_delta = self.calibrate_grasp_compute_sectors()
        print("--------------------")
        print("grasp_history:")
        for i, [grasp, grasp_delta, result] in enumerate(self.manual_grasp_calibration_history):
          print(i, grasp)
          # self.robot_coordinates.append((pose.x, pose.y, pose.z))
          self.robot_coordinates.append((grasp[0] + grasp_delta[0], grasp[1] + grasp_delta[1], grasp[2] + grasp_delta[2]))
        print("--------------------")
        print("grasp_delta_history:")
        for i, [grasp, grasp_delta, result] in enumerate(self.manual_grasp_calibration_history):
          print(i, grasp_delta)
          self.camera_coordinates.append((grasp[0], grasp[1], grasp[2]))
        print("--------------------")
        print("sector_grasp_delta:")
        print(self.sector_grasp_delta)
        print("--------------------")

        sect_cnt = 0
        if self.sector_grasp_delta != None:
          for sect in range(SECTOR_SIZE*SECTOR_SIZE):
            if self.sector_grasp_delta[sect][4] > 0:
              sect_cnt += 1
            elif sect_cnt > SECTOR_SIZE*SECTOR_SIZE/2:
              print("sect", sect," not yet calibrated") 
        if sect_cnt == (SECTOR_SIZE*SECTOR_SIZE):
          self.calibration_matrix2, self.residuals2 = self.compute_calibration(
            executor.robot_coordinates, executor.camera_coordinates)

    def set_calibration(self, T_F):
        self.calibration_mode = T_F

    def calibrate_grasp(self, grasps, confidences = None, policy = None, manual_label=False):
        try:
            action = "GRASP"
            # set values for initial grasp
            cur_pose =  self.get_pose()
            [cur_x, cur_y, cur_z] = [cur_pose[0], cur_pose[1], cur_pose[2]]
            e_i = 1
            visited_c_id = []
            for grasp in grasps:
              if grasp == None:
                continue
              [new_x, new_y, new_z, new_theta] = grasp
              print("grasps:",grasp)
              print("grasp: ",new_x, new_y, new_z, new_theta)
              c_id = policy.get_grasp_cluster(grasp)
              # if c_id == None or c_id in visited_c_id:
              #   continue
              # else:
              #   visited_c_id.append(c_id)
            
              prelift_z = min(PRELIFT_HEIGHT, (new_z - GRIPPER_OFFSET - .02))
              lift_z = new_z - GRIPPER_OFFSET
              if (action == "GRASP"):                     
                try:
                  # Start from neutral; May choose different grasp/cluster
                  # while calibrating, out of bounds may actually be in-bounds
                  # assert inside_polygon( (new_x, new_y, new_z), 
                  #   END_EFFECTOR_BOUNDS), 'Grasp not in bounds'
                  assert self.widowx.orient_to_pregrasp(
                      new_x, new_y), 'Failed to orient to target'
                  self.record_action(action,"orient_to_pregrasp",
                      [["GOAL_X", new_x],["GOAL_Y", new_y]])
                except Exception as e:
                      print('Error executing grasp -- skipping grasp...')
                      traceback.print_exc(e)
                      continue
                [delta_x, delta_y, delta_z, delta_theta, new_action] = [0, 0, 0, 0, None] 
                grasp_delta = [delta_x, delta_y, delta_z, delta_theta, new_action] 
    
                #############
                while True:
                  calib_new_x = new_x + delta_x
                  calib_new_y = new_y + delta_y
                  calib_new_z = prelift_z + delta_z
                  calib_new_theta = new_theta + delta_theta
                  print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                        % (calib_new_x, calib_new_y, calib_new_z, calib_new_theta))
                  try:
                    assert self.widowx.move_to_grasp(calib_new_x, calib_new_y, calib_new_z, calib_new_theta), \
                      'Failed to reach pre-lift pose'

                    grasp_delta = self.manual_grasp_calibration(grasp_delta)
                    [delta_x, delta_y, delta_z, delta_theta, new_action] = grasp_delta
                  except Exception as e:
                      print('Error executing grasp -- skipping grasp...')
                      traceback.print_exc(e)
                      new_action = "SKIP_GRASP"
               
                  if new_action == "SKIP_GRASP":
                    break
                  if new_action == "NEXT_IMAGE":
                    self.manual_grasp_calibration_print_history()
                    return True, 1
                  elif new_action == "EXECUTE_GRASP":
                    break
                if new_action == "SKIP_GRASP":
                  self.manual_grasp_calibration_print_history()
                  continue

                #############
                # ARD: choose new grasp from different cluster
                #############
    
                # assert self.widowx.move_to_grasp(
                #     calib_new_x, calib_new_y, calib_new_z, new_theta), 'Failed to execute grasp'
                # self.record_action(action,"move_to_grasp",
                #     [["GOAL_X", new_x],["GOAL_Y", new_y],["GOAL_Z", prelift_z],["GOAL_THETA", new_theta]],True)
                calib_new_z = lift_z + delta_z
                reached = self.widowx.move_to_vertical(calib_new_z)
                self.record_action(action,"move_to_vertical",[["GOAL_Z", calib_new_z]])

                policy.set_target_grasp([new_x, new_y, new_z, new_theta], action)
                self.widowx.close_gripper()
                self.record_action(action,"close_gripper")
                pose =  self.get_pose()
                joints = self.widowx.get_joint_values()
                # grasps in eval_grasp_action expects confidences
                eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints)
                self.manual_grasp_calibration_history.append([grasp, grasp_delta, eval_grasp_action['EVA_SUCCESS']])
                if eval_grasp_action['EVA_SUCCESS']:
                  print("grasp successful based upon gripper servo", eval_grasp_action['EVA_CLOSURE'])
                else:
                  print("grasp unsuccessful based upon gripper servo", eval_grasp_action['EVA_CLOSURE'])

                self.manual_grasp_calibration_print_history()
                print("move_to_vert")
                reached = self.widowx.move_to_vertical(prelift_z)
                # print("get_octomap_pc")
                self.widowx.open_gripper()
                self.widowx.move_to_neutral()
                self.record_action(action,"move_to_neutral")
                eval_world_action = policy.evaluate_world()
            return True, 1
        except Exception as e:
            print('Error executing grasp -- returning...')
            traceback.print_exc(e)
            return False, 1

    #############
    # execute_goal_plan()
    #############
    # After analyzing an octomap and comparing it to the goal, a sequence
    # of actions were planned out for different clusters. This executes the
    # actions. 
    #############
    def execute_goal_plan(self, goal_plan):
        x,y,z,th = 0,1,2,3
        pick_result = []
        action_completed = []
        for goal_plan_info in goal_plan:
          try:
              gs_w_cid    = goal_plan_info[0]
              gs_g_cid    = goal_plan_info[1]
              gs_desc     = goal_plan_info[2]
              goal_info   = goal_plan_info[3]
              action_info = goal_plan_info[4]
              w_cid       = action_info[0]
              action      = action_info[1]
              print(goal_plan_info)
              if action == "PICK_PLACE":
                [w_cid, action_name, grasp, place] = action_info
                self.widowx.open_gripper()
                assert self.widowx.orient_to_pregrasp(
                    grasp[x], grasp[y]), 'Failed to orient to target'
                prelift_z = min(PRELIFT_HEIGHT, (grasp[z] - GRIPPER_OFFSET - .02))
                assert self.widowx.move_to_grasp(grasp[x], grasp[y],
                     prelift_z, grasp[th]), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(grasp[z])
                self.widowx.close_gripper()
                pose = self.get_pose()
                joints = self.widowx.get_joint_values()
                print("joints:",joints)
                gripper_gap = joints[0] - np.array(GRIPPER_CLOSED[0])
                threshold=.0003
                if (gripper_gap > threshold):
                  print("eval_grasp:", gripper_gap, joints[0], pose[0])
                  pick_result.append[w_cid, True]
                else:
                  pick_result.append[w_cid, False]
                reached = self.widowx.move_to_vertical(prelift_z)
                assert self.widowx.move_to_place(place[x], place[y],
                     prelift_z, grasp[th]), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(place[z])
                self.widowx.open_gripper()
                reached = self.widowx.move_to_vertical(prelift_z)

              elif action == "PUSH_FROM_EDGE":
                [w_cid, action_name, side, pt0, pt1, pt2, theta, gw] = action_info
                prelift_z = min(PRELIFT_HEIGHT, (pt0[z] - GRIPPER_OFFSET - .02))
                self.widowx.open_gripper(gripper_value = gw)
                assert self.widowx.orient_to_pregrasp(
                    pt0[x], pt0[y]), 'Failed to orient to target'
                assert self.widowx.move_to_grasp(pt0[x], pt0[y], prelift_z, theta), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(pt0[z])
                assert self.widowx.move_to_grasp(pt1[x], pt1[y], pt1[z], theta), \
                    'Failed to reach pre-lift pose'
                assert self.widowx.move_to_grasp(pt2[x], pt2[y], pt2[z], theta), \
                    'Failed to reach pre-lift pose'
                prelift_z = min(PRELIFT_HEIGHT, (pt2[z] - GRIPPER_OFFSET - .02))
                reached = self.widowx.move_to_vertical(pt0[z])

              elif action == "ROTATE":
                [w_cid, action_name, top_ctr_pt, theta, gw, rads] = action_info
                self.widowx.open_gripper(gripper_value = gw)
                assert self.widowx.orient_to_pregrasp(
                    top_ctr_pt[x], top_ctr_pt[y]), 'Failed to orient to target'
                prelift_z = min(PRELIFT_HEIGHT, (top_ctr_pt[z] - GRIPPER_OFFSET - .02))
                assert self.widowx.move_to_grasp(top_ctr_pt[x], top_ctr_pt[y], prelift_z, th), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(top_ctr_pt[z])
                self.widowx.wrist_rotate(rads)
                self.widowx.open_gripper()
                reached = self.widowx.move_to_vertical(prelift_z)

              elif action == "NUDGE" or action == "PUSH":
                [w_cid, action_name, start_pt, end_pt, theta, gw] = action_info
                assert self.widowx.orient_to_pregrasp(
                    start_pt[x], start_pt[y]), 'Failed to orient to target'
                prelift_z = min(PRELIFT_HEIGHT, (start_pt[z] - GRIPPER_OFFSET - .02))
                self.widowx.open_gripper(gripper_value = gw)
                assert self.widowx.move_to_grasp(start_pt[x], start_pt[y], prelift_z, theta), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(start_pt[z])
                assert self.widowx.move_to_grasp(end_pt[x], end_pt[y], end_pt[z], theta), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(end_pt[z])

              action_completed.append([w_cid, True])

          except Exception as e:
            print('Error executing action...')
            traceback.print_exc(e)
            action_completed.append([w_cid, False])

          # move to safe location and take image for visual-servoing training
          premove_z = PRELIFT_HEIGHT - GRIPPER_HEIGHT * 2
          reached = self.widowx.move_to_vertical(premove_z)
          # move to neutral so that we can take clean training image
          self.widowx.move_to_neutral()

        # goal.evalaluate_move(goal_plan, pick_result, action_completed)
        return pick_result, action_completed


    #############
    # execute_grasp(), execute_action()
    #############
    # for non-goal non-move based execution of grasps.
    # The goal was analyze-octomap, execute single action, 
    # possibly analyze after partial move (e.g.,pre-lift, after grasp attempt,
    # or post-lift). Then choose a follow-up action like drop or push.
    # Eventually, analyze became so expensive, that a goal-based move-based
    # approach became favored.
    #############
    def execute_grasp(self, grasp, grasps = None, confidences = None, policy = None, manual_label=False):
        action = ["GRASP", grasp, grasps, confidences, manual_label]
        self.execute_action(action)

    def execute_action(self, action_info):
        try:
            action = action_info[0]
            next_action = None
            if action == "PICK_PLACE":
              # [PICK_PLACE, [pick x,y,z,theta][place x,y,z,theta]]
              next_action = action
              next_action = "PLACE"
              action = "PLACE"
            if action == "GRASP":
              [action_name, grasp, grasps, confidences, manual_label] = action
              # set values for initial grasp
              new_x, new_y, new_z, new_theta = grasp
              grasp_hist = [grasp]
            cur_pose =  self.get_pose()
            [cur_x, cur_y, cur_z] = [cur_pose[0], cur_pose[1], cur_pose[2]]
            e_i = 1
            while action in ["GRASP","RETRY_GRASP", "PICKUP","EVAL_GRASP", "EVAL_WORLD_ACTION", "ROTATE", "FLIP"]:
              prelift_z = min(PRELIFT_HEIGHT, (new_z - GRIPPER_OFFSET - .02))
              lift_z = new_z - GRIPPER_OFFSET
              # if self.base_z != None and len(self.base_z) != 0:
              #   sect = get_sector(new_x,new_y)
              #   lift_z = min(new_z, self.base_z[sect]) - GRIPPER_OFFSET
                #    ('lift z', 0.43801, 0.47103, 0.4812742508453974, 0.03302)

              #   print("lift z", lift_z,new_z, self.base_z[sect],GRIPPER_OFFSET)
              # else:
              #   print("WARNING: base_z = 0")

              if (action == "GRASP"):                     
                # Start from neutral; May choose different grasp/cluster
                print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                      % (new_x, new_y, new_z, new_theta))
                assert inside_polygon( (new_x, new_y, new_z), 
                    END_EFFECTOR_BOUNDS), 'Grasp not in bounds'
                assert self.widowx.orient_to_pregrasp(
                    new_x, new_y), 'Failed to orient to target'
                self.record_action(action,"orient_to_pregrasp",
                    [["GOAL_X", new_x],["GOAL_Y", new_y]])
    
                assert self.widowx.move_to_grasp(new_x, new_y, prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action(action,"move_to_grasp",
                    [["GOAL_X", new_x],["GOAL_Y", new_y],["GOAL_Z", prelift_z],["GOAL_THETA", new_theta]])
    
                assert self.widowx.move_to_grasp(
                    new_x, new_y, lift_z, new_theta), 'Failed to execute grasp'
                self.record_action(action,"move_to_grasp",
                    [["GOAL_X", new_x],["GOAL_Y", new_y],["GOAL_Z", prelift_z],["GOAL_THETA", new_theta]],True)
                policy.set_target_grasp([new_x, new_y, new_z], action)
                self.widowx.close_gripper()
                self.record_action(action,"close_gripper")
                pose =  self.get_pose()
                joints = self.widowx.get_joint_values()
                eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints)
                if eval_grasp_action['EVA_SUCCESS']:
                  print("grasp successful based upon gripper servo", eval_grasp_action['EVA_CLOSURE'])
                else:
                  print("grasp unsuccessful based upon gripper servo", eval_grasp_action['EVA_CLOSURE'])

                # print("move_to_vert")
                # reached = self.widowx.move_to_vertical(prelift_z)
                # print("get_octomap_pc")
                # octomap, header = self.get_octomap_pc()
                # print("eval grasp target")
                # eval_grasp_action = policy.evaluate_grasp_target(octomap, grasp)
                # if eval_grasp_action['EVA_SUCCESS']:
                  # print("grasp target moved")
                # else:
                  # print("grasp target unmoved")
              elif (action == "RETRY_GRASP"):                     
                # do not go all the way to neutral; facilitate digital servoing
                reached = self.widowx.move_to_vertical(prelift_z)
                self.record_action(action,"move_to_vertical",[["GOAL_Z", prelift_z]])
                assert self.widowx.move_to_grasp(new_x, new_y, 
                     prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                reached = self.widowx.move_to_vertical(new_z)
                self.widowx.close_gripper()
                pose = self.get_pose()
                joints = self.widowx.get_joint_values()
                # gripper_gap = joints[0] - np.array(GRIPPER_CLOSED[0])
                # threshold=.0003
                # if (gripper_gap > threshold):
                  # print("eval_grasp:", gripper_gap, joints[0], pose[0])
                  # pick_result = True
                # else:
                  # pick_result = False
                # reached = self.widowx.move_to_vertical(prelift_z)
                eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints)
                if eval_grasp_action['EVA_SUCCESS']:
                  print("grasp successfull based upon gripper servo", eval_grasp_action['EVA_CLOSURE'])
                reached = self.widowx.move_to_vertical(prelift_z)
                octomap, header = self.get_octomap_pc()
                eval_grasp_action = policy.evaluate_grasp_target(octomap, grasp)
              elif (action == "PICKUP"):                     
                assert self.widowx.move_to_grasp(new_x, new_y, 
                     prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action(action,"move_to_grasp",
                    [["GOAL_X", new_x],["GOAL_Y", new_y],["GOAL_Z", prelift_z],["GOAL_THETA", new_theta]])

              elif (action == "EVAL_WORLD_ACTION"):                     
                # Go to neutral pose to better determine amount of change 
                # & next action
                # self.widowx.move_to_neutral()
                # self.record_action(action,"move_to_neutral")
                eval_world_action = policy.evaluate_world()

                # GRASP OR DROP ONLY OPTION

              elif (action == "ROTATE"):                     # rotate 20 degrees
                self.widowx.wrist_rotate(DEG20_IN_RADIANS)
                self.record_action("ROTATE","wrist_rotate",
                      [["RADS", DEG20_IN_RADIANS]])
                self.widowx.open_gripper()
                self.record_action("ROTATE","open_gripper")

              elif (action == "FLIP"):         
                assert self.widowx.orient_to_pregrasp(
                  new_x, new_y), 'Failed to orient to target'
                self.record_action("FLIP","orient_to_pregrasp",
                      [["GOAL_X", new_x]["GOAL_Y", new_y]])
                assert self.widowx.move_to_grasp(new_x, new_y, 
                     prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action("FLIP","move_to_grasp",
                      [["GOAL_X", new_x]["GOAL_Y", new_y],["GOAL_Z", prelift_z]["GOAL_THETA", new_theta]])
                reached = self.widowx.move_to_vertical(new_z)
                self.record_action(action,"move_to_vertical", [["GOAL_Z", new_z]])
                # Flips 90 degrees and drops
                assert self.widowx.flip_and_drop(), 'Failed to flip and drop'
                print("FLIP","flip_and_drop")
                assert self.widowx.move_to_grasp(
                    new_x, new_y, lift_z, new_theta), 'Failed to execute grasp'
                self.record_action("FLIP","move_to_grasp",
                    [["GOAL_X", new_x]["GOAL_Y", new_y],["GOAL_Z", lift_z]["GOAL_THETA", new_theta]], True)
                self.widowx.open_gripper()
                self.record_action("FLIP","open_gripper",do_print=True)

              e_i += 1
              pose = self.get_pose()
              joints = self.widowx.get_joint_values()
              print("joints:",joints)
              # TODO: the following are for recording training data
              eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints, e_i)
              action = eval_grasp_action['EVA_ACTION']
              new_x, new_y, new_z, new_theta = eval_grasp_action['EVA_NEW_POSE']
              cur_x, cur_y, cur_z, cur_theta = eval_grasp_action['EVA_POSE']
              # current simplification
              if next_action == None:
                action = "RANDOM_DROP"   # ARD
              else:
                action = next_action
                
              # end grasping loop

            # self.widowx.move_to_neutral()
            # ARD: todo: post_move_sample
            # post_move_sample('after')
            eval_world_action = policy.evaluate_world()

            # GRAB COMPLETE; DO DROP IF SUCCESSFUL
            # action = eval_world_action['EVA_ACTION']
            # action = "RANDOM_DROP"
            if action in ["NO_DROP","PUSH","PUSH_FROM_EDGE","RANDOM_DROP","AIMED_DROP","ISOLATED_DROP","PLACE"]:
              def pos_neg():
                if random() >=.5:
                  return -1
                return 1
              if (action == "NO_DROP"):    
                pass
              elif (action == "PUSH_AFTER_FAILED_GRASP"):    
                # ARD: TODO
                # Sine: sin(theta) = Opposite / Hypotenuse
                # Cosine: cos(theta) = Adjacent / Hypotenuse
                # Tangent: tan(theta) = Opposite / Adjacent
                joints = self.widowx.get_joint_values()
                cur_angle = joints[0]
                delta = np.pi / 8
                if cur_angle + delta > np.pi:
                  new_angle = cur_angle - delta
                elif cur_angle - delta < -np.pi:
                  new_angle = cur_angle + delta
                elif random() >=.5:
                  new_angle = cur_angle - delta
                else:
                  new_angle = cur_angle + delta

                assert self.widowx.orient_to_pregrasp(angle = new_angle), 'Failed to orient to target'
                self.record_action("PUSH", "open_gripper")

                reached = self.widowx.move_to_vertical(new_z)
                self.record_action("PUSH","move_to_vertical", [["GOAL_Z", new_z]])
                self.widowx.move_to_neutral()
                # ARD: todo: post_move_sample
                # post_move_sample('after')
                # eval_world_action = policy.evaluate_world()
              elif (action == "PUSH_FROM_EDGE"):         # rotate 20 degrees
                pass
              elif (action == "FLIP_DROP"):         # rotate 20 degrees
                assert self.widowx.orient_to_pregrasp(
                  x, y), 'Failed to orient to target'
                self.record_action("FLIP_DROP","orient_to_pregrasp",
                      [{"GOAL_X", new_x},{"GOAL_Y", new_y}])
                prelift_z = min(PRELIFT_HEIGHT, (z - GRIPPER_OFFSET - .02))
                assert self.widowx.move_to_grasp(x, y, prelift_z, theta), \
                    'Failed to reach pre-lift pose'
                self.record_action("FLIP_DROP","move_to_grasp",
                    [{"GOAL_X", new_x},{"GOAL_Y", new_y},{"GOAL_Z", prelift_z},{"GOAL_THETA", new_theta}])
                # Flips 90 degrees and drops
                assert self.widowx.flip_and_drop(), 'Failed to flip and drop'
                self.record_action("FLIP_DROP","flip_and_drop")
                assert self.widowx.move_to_grasp(
                    x, y, z, theta), 'Failed to execute grasp'
                self.record_action("FLIP_DROP","move_to_grasp",
                    [{"GOAL_X", new_x},{"GOAL_Y", new_y},{"GOAL_Z", prelift_z},{"GOAL_THETA", new_theta}], True)
                self.widowx.open_gripper()
                self.record_action(action, "open_gripper")
              else:
                # default z is to place near base
                x = new_x
                y = new_y
                z = new_z
                theta = new_theta
# ARD: BASE_Z
                if self.base_z != None and len(self.base_z) != 0:
                  # base_z is part of WorldState
                  sect = get_sector(x,y)
                  z_at_base = min(z, self.base_z[sect])
                else:
                  z_at_base = z
                z_at_base -= GRIPPER_OFFSET
                if (action == "RANDOM_DROP"):
                  x = random() * self.drop_eeb[0] * pos_neg()
                  y = random() * self.drop_eeb[1] * pos_neg()
                  z = prelift_z   
                elif (action == "AIMED_DROP"):
                  x,y,z = interesting_cluster()
                elif action == "ISOLATED_DROP":
                  x,y,z = unoccupated_space()
                elif action == "PLAYGROUND_DROP":
                  x,y,z = unoccupated_playground_space()
          
                assert self.widowx.move_to_grasp(x, y, prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action(action,"move_to_grasp",
                    [{"GOAL_X", new_x},{"GOAL_Y", new_y},{"GOAL_Z", prelift_z},{"GOAL_THETA", new_theta}])
                assert self.widowx.move_to_grasp(
                    x, y, z, theta), 'Failed to drop'
                self.record_action(action,"move_to_grasp",
                    [{"GOAL_X", new_x},{"GOAL_Y", new_y},{"GOAL_Z", prelift_z},{"GOAL_THETA", new_theta}], True)
                reached = self.widowx.move_to_vertical(z)
                self.record_action(action,"move_to_vertical",[{"GOAL_Z",z}])
                self.widowx.open_gripper()
                self.record_action(action,"open_gripper",True)
                rospy.sleep(2)
            elif eval_grasp_action['EVA_ACTION'] == "SWEEP_ARENA":
              # Too many failures and objects all along the side or corners
              # ARD TODO: improve sweep
              executor.widowx.sweep_arena()
              self.record_action("SWEEP_ARENA","sweep_arena")
              executor.widowx.move_to_neutral()
              self.record_action("SWEEP_ARENA","move_to_neutral")
              executor.widowx.open_gripper()
              self.record_action("SWEEP_ARENA","open_gripper")
            else:                                  # original code
              self.widowx.move_to_drop(), 'Failed to move to drop'
              self.record_action("MOVE_TO_DROP","move_to_drop")
              # rospy.sleep(2)
              success = policy.evaluate_grasp(manual=manual_label)
              self.sample['gripper_closure'] = self.widowx.eval_grasp()[1]

            self.widowx.move_to_neutral()
            self.record_action("END_GRASP","move_to_neutral")
            success = policy.evaluate_world()
            return success, 0

        except Exception as e:
            print('Error executing grasp -- returning...')
            traceback.print_exc(e)
            return False, 1

    def calculate_crop(self, grasp):
        grasp = np.concatenate([grasp, [1.]], axis=0)
        transformedPoint = np.dot(self.inv_cm, grasp)
        predicted = self.camera.project3dToPixel(transformedPoint)
        return int(predicted[0]), int(predicted[1])

    def compute_calibration(self, robot_points, camera_points):
        lr = LinearRegression().fit(camera_points, robot_points)
        predicted = lr.predict(camera_points)
        residuals = np.abs(predicted - robot_points)
    
        co = lr.coef_
        trans = lr.intercept_
        tf_matrix = np.matrix([[co[0, 0],	co[0, 1],	co[0, 2],	trans[0] ],
                               [co[1, 0],	co[1, 1],	co[1, 2],	trans[1] ],
                               [co[2, 0],	co[2, 1],	co[2, 2],	trans[2] ],
                               [0.0,		0.0,		0.0,		1.0		]])

        print('Residuals (cm):')
        print(residuals * 100.)
        print('Calibration matrix:')
        print(calibration_matrix)

        return tf_matrix, residuals


    def publish_grasps(self, grasps = None, chosen_grasp = None):
        if not DISPLAY_PC_GRASPS:
          return
        pc2 = []
        if grasps is not None:
          for i, (g, prob) in enumerate(grasps):
            # rgba = (cluster+1) * 4294967294 / 2 / (nm_clusters+1)
            # convert degrees (g[3]) to color
            deg = g[3]
            if deg < 0:
              deg = deg + 360
            # print("deg ",deg)
            rgba = int(deg * 4294967294 / 2 / 360 + 255)
            # print("rgba ",rgba)
            # 4294967040 = FFFFFF00
            if chosen_grasp is not None:
              if chosen_grasp[0] == g[0] and chosen_grasp[1] == g[1] and chosen_grasp[2] == g[2]:
                # bright yellow =  rgb(255,255,0) transparency = 0
                # rgba = 4294901760
                WHITE = 4294967294 / 2
                rgba = WHITE
                print("Chosen grasp: ", chosen_grasp[0], chosen_grasp[1], chosen_grasp[2])
            p2 = [g[0],g[1],g[2],rgba]
            pc2.append(p2)
        # if FAVOR_KEYPOINT:
        #   pc2 = KP.add_to_pc(pc2)

        pc2 = np.reshape(pc2, (len(pc2), 4))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        pc2 = point_cloud2.create_cloud(self.pc_header, fields, pc2)
        self.pc_grasps_pub3.publish(pc2)

    def publish_pc(self, topic, pc):
        # Derived from:
        # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
        pc = np.reshape(pc, (len(pc), 4))
        pc = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc]
        print("pc len: ", len(pc))
        if len(pc) > 0:
          # print("pc[0]: ", pc[0])
          pass
        else:
          return
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        if topic == PC_RGB_TOPIC:
          pc = point_cloud2.create_cloud(self.pc_header, fields, pc)
          self.pc_rgb_pub1.publish(pc)
        elif topic == PC_OCTOMAP_TOPIC:
          fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]

          pc = point_cloud2.create_cloud(self.octomap_header, fields, pc)
          self.pc_octomap_pub2.publish(pc)

    def record_action(self, action_name, subaction_name, action_data=None, do_print=False):
      # todo: change to store octomap
      self.history.new_action( [["ACTION",action_name, subaction_name], 
         ["POSE", self.get_pose()],["JOINTS",self.widowx.get_joint_values()]])
         # ["RGBD", self.get_rgbd()], 
      if action_data != None:
        self.history.action_info(action_data)
      if do_print != None and do_print:
        print(action_name,subaction_name)

    def record_grasp(self, grasp, grasps, calibration_adjustment = None):
      self.history.new_event()
      self.history.new_action("EXECUTE_GRASP","execute_grasp")
      self.history.action_info(["CHOSEN_GRASP",grasp[0],grasp[1],grasp[2],grasp[3]])
      for i, (grasp, confidence) in enumerate(grasps):
        # i, x, y, z, theta, confidence
        self.history.action_info(["GRASP", i, grasp[0],grasp[1],grasp[2],grasp[3],confidence,calibration_adjustment])

