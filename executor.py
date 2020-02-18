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

    def __init__(self, scan=False, datapath='', save=False):
        # WidowX controller interface
        self.widowx = WidowX()
        self.history = ActionHistory()
        self.transform = lambda x: x
        # For ROS/cv2 conversion
        self.bridge = CvBridge()
        self.rgb_ready = True
        self.pc_ready = True
        self.pc_count = 0

        # Register subscribers
        if not RGB_DEPTH_FROM_PC:
          self.img_subscriber = rospy.Subscriber(
            RGB_IMAGE_TOPIC, Image, self.update_rgb)
          self.depth_subscriber = rospy.Subscriber(
            DEPTH_IMAGE_TOPIC, Image, self.update_depth)
          self.caminfo_subscriber = rospy.Subscriber(
            DEPTH_CAMERA_INFO_TOPIC, CameraInfo, self.save_cinfo)
        self.pc_subscriber = rospy.Subscriber(
            POINTCLOUD_TOPIC, PointCloud2, self.update_pc)

        # Register cluster pub
        if DISPLAY_PC_RGB:
          self.pc_rgb_pub1 = rospy.Publisher(PC_RGB_TOPIC, PointCloud2, queue_size=1)
        if DISPLAY_PC_DEPTH_MAP:
          self.pc_depth_map_pub2 = rospy.Publisher(PC_DEPTH_MAP_TOPIC, PointCloud2, queue_size=1)
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

        if RGB_DEPTH_FROM_PC:
          self.rgb = np.zeros((RGB_HEIGHT, RGB_WIDTH, 3))
          self.depth = np.zeros((RGB_HEIGHT, RGB_WIDTH, 1))
          self.pc_map = np.full((RGB_HEIGHT, RGB_WIDTH), -1)
        else:
          self.rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
          self.depth = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
          self.pc_map = none
        # ARD: orig:
        # self.pc = np.zeros((1, 3))
        self.pc = []                   # unmaterialized pc
        self.pc_img = []                   
        self.full_pc = []
        self.full_pc_rgb = []
        self.transformed_pc = []
        self.transformed_pc_rgb = []   
        self.base_z = []
        self.sample = {}
        self.pc_header = None

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

    # rgb image topic if not RGB_DEPTH_FROM_PC
    def update_rgb(self, data):
        if not self.rgb_ready:
          print("skip update_rgb")
          return
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.rgb = cv_image

    # depth image topic if not RGB_DEPTH_FROM_PC
    def update_depth(self, data):
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.depth = cv_image

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

    def get_rgbd(self):
        self.get_pc()
        if RGB_DEPTH_FROM_PC:
          # depth = np.reshape(self.depth, (RGB_WIDTH, RGB_HEIGHT, 1))
          # depth = np.reshape(self.depth, (RGB_HEIGHT, RGB_WIDTH, 1))
          # print("   ", self.depth.shape, self.rgb.shape)
          # print("   ", len(self.pc_img), len(self.pc_img[0]))
          return self.pc_img
          # self.full_pc_rgb = np.concatenate([self.rgb, depth], axis=2)
          # return self.full_pc_rgb 
          # return self.rgbd
        else: # derived from original code
          # old_depth = self.depth.astype(np.float) / 10000.
          old_depth = self.depth.astype(np.float) 
          # ARD
          # print ("old_depth.shape ")
          # print (old_depth.shape)    # (480, 640, 1)
          depth = rectify_depth(old_depth)
          depth = np.reshape(depth, (IMG_HEIGHT, IMG_WIDTH, 1))
          self.full_pc_rgb = np.concatenate([self.rgb, depth], axis=2)
          return self.full_pc_rgb 

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

    def get_pc(self):
        # ARD
        # while (not self.pc_ready):
        #   yield
        # pc = list(self.pc)
        # if len(self.pc) == 0:
        #     return 
        # self.pc_ready = False
        # pc = list(self.pc)
        # self.pc_ready = True

        # pc = np.array(self.pc)[:, :3]
        # materialize pc
        self.pc_ready = False
        pc = list(self.pc)
        self.pc_ready = True
        # print("len self.pc ", len(pc))
        if len(pc) == 0:
          print("No point cloud from camera!")
          return None
        rgb_pc = np.array(pc)[:, :4]
        np.random.shuffle(rgb_pc)
        # print("rgb_pc shape0 ", rgb_pc.shape[0])
        print("rgb_pc shape ", rgb_pc.shape)
        # if rgb_pc.shape[0] > RGB_PC_DENSITY:
        # After filtering out Tray, RGb_PC_DENSTITY is not sufficient
        # if len(rgb_pc) > RGB_PC_DENSITY:
        #   rgb_pc = rgb_pc[:RGB_PC_DENSITY]
        #   print("len rgb_pc ", len(rgb_pc))
        pc = np.array(rgb_pc)[:, :3]

        def transform_pc(srcpc, tf_matrix):
            ones = np.ones((srcpc.shape[0], 1))
            srcpc = np.append(srcpc, ones, axis=1)
            out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc = transform_pc(pc, self.cm)

        # put rgb back in pc
        # print("b4 len pc ", len(pc))
        rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS)]
        pc = rgb_pc
        # pc = [p for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS)]
        # rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS)]
        # rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc)]
        # pc = [p for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS)]
        print("len pc ", len(pc))
        # if len(pc) > 0:
          # print("pc[0]: ",pc[0])
        if RGB_DEPTH_FROM_PC:
          # get 2d rgb image, with associated depth, and a mapping to full pc
          self.rgb, self.depth, self.pc_map, self.pc_img = rgb_depth_map_from_pc(pc,rgb_pc)
          self.full_pc = pc
          self.sample['full_pc'] = self.full_pc

        ######################
        # Filter out base tray
        ######################
        print("prefilter pc len:",len(pc))
        self.base_z = compute_z_sectors(pc)
        # ARD: any (base_z - MIN_OBJ_HEIGHT) greater than .485 is suspect...
        # print(MIN_OBJ_HEIGHT, " sector base_z:", self.base_z)
        pc2_rgb = []
        for i, p in enumerate(pc):
             sect = get_sector(p[0],p[1])
             if (p[2] > self.base_z[sect] - MIN_OBJ_HEIGHT):
               # if i < 10:
                 # print("TOO CLOSE TO GROUND ", p[2], self.base_z[sect], MIN_OBJ_HEIGHT)
               continue
             if (p[2] > 0.485):
               print("suspect sector: ", sect, self.base_z[sect], p)
             pc2_rgb.append(p)
        pc = pc2_rgb
        print("postfilter pc len:",len(pc))
        ######################

        # ARD: TEST
        # self.transformed_pc_rgb = self.pc_img
        # self.publish_pc()
        # print(jojo)
        # ARD: END TEST
        # do further PC size reduction for kdtree and grip analysis
        # if pc.shape[0] > PC_DENSITY:
        # pc = np.array(self.pc_img)[:, :3]
        if len(pc) > PC_DENSITY:
            pc = pc[:PC_DENSITY]
        print("pruned pc len:",len(pc))

# Moved to plan_grasp
#        dist, _ = self.kdtree.query(pc, k=1)
#        # ARD: Huge reduction of points (~5K to ~975 with dups, 670 no dups
#        # ARD: Better off using get_pc_rgb()?
#        pc = [p for i, p in enumerate(pc) if dist[i] > .003]
#        # PC_BOUNDS, HEIGH_BOUNDS now done in rgb_depth_map_from_pc()
#        # pc = [p for i, p in enumerate(pc) if inside_polygon(
#        #     p, PC_BOUNDS, HEIGHT_BOUNDS) and dist[i] > .003]

        # pc = np.reshape(pc, (len(pc), 3))
        pc = np.reshape(pc, (len(pc), 4))
        self.transformed_pc_rgb = pc
        # for i, p in enumerate(pc):
          # print(i,"pc",pc[i])
        return pc

    # return img
    def get_rgb(self):
        return self.rgb
        # return cv2.UMat(self.rgb)
        # return cv2.cvtColor(self.rgb, RGB)
        # import scipy.misc
        # return scipy.misc.toimage(self.rgb)
        # from PIL import Image
        # return Image.fromarray(self.rgb, 'RGB')

    # 2-d image of clusters (i.e., tray filtered out)
    def get_pc_rgb(self):
        if RGB_DEPTH_FROM_PC:
          # self.full_pc_rgb = self.full_pc[:, 3]
          # self.full_pc_rgb = list(self.pc)
          # executor.get_rgbd() and get_pc() should have already been called:
          # -> self.rgb, self.depth, self.pc_map, self.full_pc, self.full_pc_rgb
          #    are filled already in
          # print("pc[0] :",self.full_pc_rgb[0])
          # self.full_pc_rgb = np.array(self.full_pc_rgb)[:, :4]
          # print("fpc ",len(self.full_pc))
          # print("fpc2",self.full_pc[0].shape)
          # self.full_pc_rgb = np.array(self.full_pc)[:, :4]
          print("pc_img ",len(self.pc_img))    # 16320 == (120x136)
          pc = self.pc_img
        else:
          ###################
          # ARD: No longer used
          #vvvvvvvvvvvvvvvvvv
          # while (not self.pc_ready):
          #   yield
          prev_pc_cnt = -1
          # different light structures get different results
          while not self.pc_ready or prev_pc_cnt == self.pc_count:
            print("not ready")
            rospy.sleep(.1)
          prev_pc_cnt = self.pc_count
          self.pc_ready = False
          self.full_pc = list(self.pc)
          self.pc_ready = True
          prev_pc_len = len(self.full_pc) 
          # print("pc len: ", len(self.full_pc))
          self.full_pc = np.array(self.full_pc)[:, :4]
          # "atomically" evaluate generator
          np.random.shuffle(self.full_pc)
          if self.full_pc.shape[0] > PC_DENSITY:
              self.full_pc = self.full_pc[:PC_DENSITY]
          self.full_pc = np.array(self.full_pc)[:, :4]
          # self.full_pc_rgb = self.full_pc[:, 3]
          self.full_pc_rgb = self.full_pc
          # print("pc_rgb: ", self.full_pc_rgb)
          self.full_pc = np.array(self.full_pc)[:, :3]
  
          def transform_pc(srcpc, tf_matrix):
              ones = np.ones((srcpc.shape[0], 1))
              srcpc = np.append(srcpc, ones, axis=1)
              out = np.dot(tf_matrix, srcpc.T)[:3]
              return out.T
  
          self.full_pc = transform_pc(self.full_pc, self.cm)
          self.full_pc = np.array(self.full_pc)[:, :3]
          self.sample['full_pc'] = self.full_pc
  
          # Didn't like the plane results of segment cluster 
          # plus the plane logic removed some of the flat objects.
          # plane = self.segment_cluster(pc)
          # pc3 = [p for i, p in enumerate(pc) if dist[i] > .003 and
          #       (plane is not None and not plane[i])]
          pc = []
          if DISPLAY_PC_RGB:
            pc_rgb = []
            for i, p in enumerate(self.full_pc):
              if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS):
                pc.append(p)
                if DISPLAY_PC_RGB:
                  pc_rgb.append(p)
            print("len full_pc: ",len(self.full_pc), " vs. inbound ", len(pc_rgb))
          #vvvvvvvvvvvvvvvvvv
          # ARD: End No longer used
          ###################

        ######################
        # Filter out base tray
        ######################
        # prefilter: 16320.  Post-filter: 965, with duplicates
        self.base_z = compute_z_sectors(pc)
        pc2 = []
        if DISPLAY_PC_RGB:
          pc2_rgb = []
          for i, p in enumerate(pc):
             sect = get_sector(p[0],p[1])
             if (p[2] > self.base_z[sect] - MIN_OBJ_HEIGHT):
               # if i < 10:
                 # print("TOO CLOSE TO GROUND ", p[2], self.base_z[sect], MIN_OBJ_HEIGHT)
               continue
             pc2.append(p)
             # pc2_rgb.append(pc_rgb[i])
             pc2_rgb.append(p)
        # pc = np.reshape(pc2, (len(pc2), 3))
        if DISPLAY_PC_RGB:
          # self.transformed_pc_rgb = np.reshape(pc2_rgb, (len(pc2_rgb), 3))
          self.transformed_pc_rgb = np.reshape(pc2_rgb, (len(pc2_rgb), 4))
          # self.transformed_pc = pc
        # print("pc: ", self.pc)
        # print("pc_rgb: ", self.pc_rgb)
        # print("len1 pc inbounds: ",len(self.pc), " pc[0]= ", self.pc[0])
        # return self.pc, self.pc_rgb
        # return self.transformed_pc, self.transformed_pc_rgb

        # ARD: Used by plan_grasp()
        return self.transformed_pc_rgb

    def scan_base(self, scans=100):
        def haul(pc):
            try:
                rgbd = executor.get_rgbd()
                pc = self.get_pc()
                print('# of new base points: %d' % len(pc))
                self.base_pc = np.concatenate([self.base_pc, pc], axis=0)
                np.save(PC_BASE_PATH, self.base_pc)
                self.kdtree = KDTree(self.base_pc)
            except ValueError as ve:
                traceback.print_exc(ve)
                print('No pointcloud detected')

        for i in range(scans):
            print('Scan %d' % i)
            haul(self.pc)
            rospy.sleep(1)

    def execute_grasp(self, grasp, grasps = None, confidences = None, policy = None, manual_label=False):
        try:
            action = "GRASP"
            # set values for initial grasp
            new_x, new_y, new_z, new_theta = grasp
            grasp_hist = [grasp]
            cur_pose =  self.get_pose()
            [cur_x, cur_y, cur_z] = [cur_pose[0], cur_pose[1], cur_pose[2]]
            e_i = 1
            while action in ["GRASP","RETRY_GRASP", "PICKUP","EVAL_WORLD_ACTION", "ROTATE", "FLIP"]:
              prelift_z = min(PRELIFT_HEIGHT, (new_z - GRIPPER_OFFSET - .02))
              if len(self.base_z) != 0:
                sect = get_sector(new_x,new_y)
                lift_z = min(new_z, self.base_z[sect]) - GRIPPER_OFFSET
                #    ('lift z', 0.43801, 0.47103, 0.4812742508453974, 0.03302)

                print("lift z", lift_z,new_z, self.base_z[sect],GRIPPER_OFFSET)
              else:
                print("WARNING: base_z = 0")

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
                self.widowx.close_gripper()
                self.record_action(action,"close_gripper")


              elif (action == "RETRY_GRASP"):                     
                # do not go all the way to neutral; facilitate digital servoing
                reached = self.widowx.move_to_vertical(prelift_z)
                self.record_action(action,"move_to_vertical",[["GOAL_Z", prelift_z]])
                assert self.widowx.move_to_grasp(new_x, new_y, 
                     prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action(action,"move_to_grasp",
                    [["GOAL_X", new_x]["GOAL_Y", new_y],["GOAL_Z", prelift_z]["GOAL_THETA", new_theta]])
                self.commander.execute(plan, wait=True)
                reached = self.widowx.move_to_vertical(new_z)
                self.record_action(action, "move_to_vertical", [["GOAL_Z", new_z]])
                self.widowx.close_gripper()
                self.record_action(action, "close_gripper")
                pose = self.get_pose()
                joints = self.widowx.get_joint_values()
                print("joints:",joints)
                eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints)

              elif (action == "PICKUP"):                     
                assert self.widowx.move_to_grasp(new_x, new_y, 
                     prelift_z, new_theta), \
                    'Failed to reach pre-lift pose'
                self.record_action(action,"move_to_grasp",
                    [["GOAL_X", new_x],["GOAL_Y", new_y],["GOAL_Z", prelift_z],["GOAL_THETA", new_theta]])

              elif (action == "EVAL_WORLD_ACTION"):                     
                # Go to neutral pose to better determine amount of change 
                # & next action
                self.widowx.move_to_neutral()
                self.record_action(action,"move_to_neutral")
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
              eval_grasp_action = policy.evaluate_grasp(grasp, grasps, pose, joints, e_i)
              action = eval_grasp_action['EVA_ACTION']
              new_x, new_y, new_z, new_theta = eval_grasp_action['EVA_NEW_POSE']
              cur_x, cur_y, cur_z, cur_theta = eval_grasp_action['EVA_POSE']
              # end grasping loop

            # self.widowx.move_to_neutral()
            # ARD: todo: post_move_sample
            # post_move_sample('after')
            eval_world_action = policy.evaluate_world()

            # GRAB COMPLETE; DO DROP IF SUCCESSFUL
            # action = eval_world_action['EVA_ACTION']
            action = "RANDOM_DROP"
            if action in ["NO_DROP","PUSH","RANDOM_DROP","AIMED_DROP","ISOLATED_DROP"]:
              def pos_neg():
                if random() >=.5:
                  return -1
                return 1
              if (action == "NO_DROP"):    
                pass
              elif (action == "PUSH"):    
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
                if len(self.base_z) != 0:
                  sect = get_sector(x,y)
                  z_at_base = min(z, self.base_z[sect])
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

    def publish_pc(self):
      # Derived from:
      # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
      for j in range(2):
        if j == 0 and not (DISPLAY_PC_DEPTH_MAP and DISPLAY_PC_RGB):
          continue
        elif j == 0:
          # pc = self.transformed_pc
          # pc = self.transformed_pc_rgb
          # self.rgb, self.depth
          print("transformed_pc ",len(self.transformed_pc_rgb))
          pc_rgb = self.transformed_pc_rgb
          # self.base_z = compute_z_sectors(self.transformed_pc_rgb)
          # pc_rgb = pc_depth_mapping(self.transformed_pc, self.base_z)
          # pc_rgb = pc_depth_mapping(self.transformed_pc_rgb, self.base_z)
          # print("pc_rgb ",len(pc_rgb))
        elif DISPLAY_PC_RGB and j == 1:
          # pc = self.full_pc
          # pc_rgb = self.full_pc_rgb
          # pc_rgb = self.transformed_pc_rgb
          pc_rgb = self.pc_img
          # print("full_pc ",len(self.full_pc))
          # print("full_pc_rgb ",len(self.full_pc_rgb))
        else:
          continue
        pc2 = pc_rgb
        # pc2 = []
        # print("len(pc): ",len(pc))
        # for i, p in enumerate(pc):
          # p2 = [p[0],p[1],p[2],pc_rgb[i][3]]
          # pc2.append(p2)
        # if FAVOR_KEYPOINT:
        #   pc2 = KP.add_to_pc(pc2)
        pc2 = np.reshape(pc2, (len(pc2), 4))
        print("pc2 len: ", len(pc2))
        print("pc_rgb len: ", len(pc_rgb))
        if len(pc2) > 0:
          print("pc2[0]: ", pc2[0])
        else:
          continue
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        pc2 = point_cloud2.create_cloud(self.pc_header, fields, pc2)
        if j == 0 and DISPLAY_PC_RGB:
          self.pc_rgb_pub1.publish(pc2)
        else:
          self.pc_depth_map_pub2.publish(pc2)

    def record_action(self, action_name, subaction_name, action_data=None, do_print=False):
      # For NN training, we will use the fixed sized self.rgbd instead of 
      # using self.full_pc():
      #     ["PC",self.get_pc()],
      #     ["RGBD", self.get_rgbd()], # ARD: Not needed for most actions
      self.history.new_action( [["ACTION",action_name, subaction_name], 
         ["RGBD", self.get_rgbd()], 
         ["POSE", self.get_pose()],["JOINTS",self.widowx.get_joint_values()]])
      if action_data != None:
        self.history.action_info(action_data)
      if do_print != None and do_print:
        print(action_name,subaction_name)

    def record_grasp(self, grasp, grasps):
      self.history.new_event()
      self.history.new_action("EXECUTE_GRASP","execute_grasp")
      self.history.action_info(["CHOSEN_GRASP",grasp[0],grasp[1],grasp[2],grasp[3]])
      for i, (grasp, confidence) in enumerate(grasps):
        # i, x, y, z, theta, confidence
        self.history.action_info(["GRASP", i, grasp[0],grasp[1],grasp[2],grasp[3],confidence])

