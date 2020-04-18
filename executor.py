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
        self.pc_ready = True
        self.pc_count = 0
        self.octomap_ready = False
        self.octomap_ready2 = False

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
        # while self.pc_count != self.octomap_count:
        #   print(" self.pc_count:", self.pc_count," octomap_count:", self.octomap_count)
        #   rospy.sleep(1)
        # filter out tray
        self.octomap_pc = []
        while len(self.octomap_pc) == 0:
          self.octomap_ready = True
          if self.octomap_ready2:
            self.octomap_pc = (self.octomap)
            # self.octomap_pc = list(self.octomap)
            # self.octomap_pc = [[p[0],p[1],p[2],np.uint32(p[3])] for p in self.octomap]
            self.octomap_ready = False
          print("len octomap_pc:",len(self.octomap_pc))
          if len(self.octomap_pc) > 0:
            print("octomap[0]:",self.octomap_pc[0])
          if len(self.octomap_pc) == 0:
            rospy.sleep(2)
        # octomap_pc = np.array(self.octomap_pc)[:, :4]
        print("1 octomap_pc[0]:",self.octomap_pc[0])
        # self.octomap_pc = self.filter_base(self.octomap_pc)
        # octomap returns average color, so no need to use color from closest point in transformed pc.
        self.octomap_pc = add_color(self.octomap_pc, self.transformed_pc_rgb)
        print("2 octomap_pc[0]:",self.octomap_pc[0])
        
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
            out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc = transform_pc(pc, self.cm)

        # put rgb back in pc
        rgb_pc = [[p[0],p[1],p[2],rgb_pc[i][3]] for i, p in enumerate(pc) if inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS)]
        pc = rgb_pc
        print("len pc ", len(pc))

        # pruning not required with Octomap. Higher res is better.
        # # do further PC size reduction for kdtree and grip analysis
        # if len(pc) > PC_DENSITY:
        #     pc = pc[:PC_DENSITY]
        # print("pruned pc len:",len(pc))

        pc = np.reshape(pc, (len(pc), 4))
        self.transformed_pc_rgb = pc
        self.publish_pc(PC_RGB_TOPIC, self.transformed_pc_rgb)
        return pc

    def filter_base(self,pc):
        ######################
        # Filter out base tray
        ######################
        
        return pc


        # z "elbow" to filter base
        return rm_base(pc)

        plane = segment_cluster(pc)
        p1 = [p for i,p in enumerate(pc) if not plane[i]]
        p2 = [p for i,p in enumerate(pc) if plane[i]]
        print("prefilter pc len:",len(pc), len(plane), len(p1), len(p2))
        pc2_rgb = p2
        print("postfilter pc len:",len(pc2_rgb))
        return pc2_rgb


        ######################
        # z "elbow" to filter base
        return rm_base(pc)

        ######################
        # use segmentation to filter base
        ######################
        print("prefilter pc len:",len(pc))
        # ARD: any (base_z - MIN_OBJ_HEIGHT) greater than .485 is suspect...
        # print(MIN_OBJ_HEIGHT, " sector base_z:", self.base_z)

        plane = segment_cluster(pc)
        pc2_rgb = []
        for i, p in enumerate(pc):
          if not plane[i]:
             pc2_rgb.append(p)
        print("postfilter pc len:",len(pc2_rgb))
        return pc2_rgb

        ######################
        # use averages to filter base
        ######################
        self.base_z = compute_z_sectors(pc, self.base_z)
        pc2_rgb = []
        for i, p in enumerate(pc):
             sect = get_sector(p[0],p[1])
             if (p[2] > self.base_z[sect] - MIN_OBJ_HEIGHT):
               if i < 10:
                 print("TOO CLOSE TO GROUND ", p[2], self.base_z[sect], MIN_OBJ_HEIGHT)
               continue
             # if (p[2] > 0.485):
             #   print("suspect sector: ", sect, self.base_z[sect], p)
             pc2_rgb.append(p)
        print("postfilter pc len:",len(pc2_rgb))
        ######################
        return pc2_rgb

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

    def publish_pc(self, topic, pc):
        # Derived from:
        # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
        pc = np.reshape(pc, (len(pc), 4))
        pc = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc]
        print("pc len: ", len(pc))
        if len(pc) > 0:
          print("pc[0]: ", pc[0])
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

