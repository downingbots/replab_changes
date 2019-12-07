#!/usr/bin/env python

import numpy as np
from matplotlib.patches import Circle
import h5py
import math
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
from keypoint import *

class Executor:

    def __init__(self, scan=False, datapath='', save=False):
        # WidowX controller interface
        self.widowx = WidowX()
        self.transform = lambda x: x
        # For ROS/cv2 conversion
        self.bridge = CvBridge()
        # self.keypoints = []
        self.rgb_ready = True
        self.pc_ready = True
        self.pc_count = 0

        # Register subscribers
        self.img_subscriber = rospy.Subscriber(
            RGB_IMAGE_TOPIC, Image, self.update_rgb)
        self.depth_subscriber = rospy.Subscriber(
            DEPTH_IMAGE_TOPIC, Image, self.update_depth)
        self.pc_subscriber = rospy.Subscriber(
            POINTCLOUD_TOPIC, PointCloud2, self.update_pc)
        self.caminfo_subscriber = rospy.Subscriber(
            DEPTH_CAMERA_INFO_TOPIC, CameraInfo, self.save_cinfo)

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
        self.rgb = np.zeros((480, 640, 3))
        self.depth = np.zeros((480, 640, 1))
        # ARD: orig:
        # self.pc = np.zeros((1, 3))
        self.pc = []                   # unmaterialized pc
        self.full_pc = []
        self.full_pc_rgb = []
        self.transformed_pc = []
        self.transformed_pc_rgb = []   
        self.base_z = []
        self.pc_header = None

        self.camera_info = CameraInfo()
        if scan:
            self.base_pc = np.zeros((1, 3)) + 5000.0  # adds dummy point
        else:
            self.base_pc = np.load(PC_BASE_PATH)
        self.cm = CALIBRATION_MATRIX
        self.inv_cm = np.linalg.inv(self.cm)
        self.sample = {}
        self.kdtree = KDTree(self.base_pc)
        self.recent_grasps = []

        rospy.sleep(2)

        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(self.camera_info)

    # rgb image topic
    def update_rgb(self, data):
        if not self.rgb_ready:
          print("skip update_rgb")
          return
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.rgb = cv_image

    # depth image topic
    def update_depth(self, data):
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.depth = cv_image

    def update_pc(self, data):
        if not self.pc_ready:
          # busy materializing from previous pointcloud.
          # skip and process next point cloud instead
          # print("skip u_pc;"),
          return
        self.pc_header = data.header
        # print("pc_header: ", self.pc_header)
        self.pc = pc2.read_points(data, skip_nans=True)
        self.pc_count = self.pc_count + 1
        if self.pc_count == 10:
          # self.pub_cluster()
          self.pc_count = 0

    def save_cinfo(self, data):
        self.camera_info = data

    def get_rgbd(self):
        # old_depth = self.depth.astype(np.float) / 10000.
        old_depth = self.depth.astype(np.float) 
        # ARD
        # print ("old_depth.shape ")
        # print (old_depth.shape)    # (480, 640, 1)
        depth = rectify_depth(old_depth)
        depth = np.reshape(depth, (480, 640, 1))
        return np.concatenate([self.rgb, depth], axis=2)

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
        self.pc_ready = False
        pc = list(self.pc)
        self.pc_ready = True
        pc = np.array(pc)[:, :3]

        # pc = np.array(self.pc)[:, :3]
        np.random.shuffle(pc)
        if pc.shape[0] > 5000:
            pc = pc[:5000]

        def transform_pc(srcpc, tf_matrix):
            ones = np.ones((srcpc.shape[0], 1))
            srcpc = np.append(srcpc, ones, axis=1)
            out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc = transform_pc(pc, self.cm)

        self.sample['full_pc'] = pc

        dist, _ = self.kdtree.query(pc, k=1)
        pc = [p for i, p in enumerate(pc) if inside_polygon(
            p, PC_BOUNDS, HEIGHT_BOUNDS) and dist[i] > .003]
        return np.reshape(pc, (len(pc), 3))


    def get_rgb(self):
        return self.rgb

    def get_pc_rgb(self):
        # ARD
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
        self.full_pc_rgb = self.full_pc[:, 3]
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
        self.base_z = compute_z_sectors(pc)
        pc2 = []
        if DISPLAY_PC_RGB:
          pc2_rgb = []
          for i, p in enumerate(pc):
             sect = get_sector(p[0],p[1])
             if (p[2] > self.base_z[sect] - MIN_OBJ_HEIGHT):
               # print("TOO CLOSE TO GROUND")
               continue
             pc2.append(p)
             pc2_rgb.append(pc_rgb[i])
        pc = np.reshape(pc2, (len(pc2), 3))
        if DISPLAY_PC_RGB:
          self.transformed_pc_rgb = np.reshape(pc2_rgb, (len(pc2_rgb), 3))
          self.transformed_pc = pc
        # print("pc: ", self.pc)
        # print("pc_rgb: ", self.pc_rgb)
        # print("len1 pc inbounds: ",len(self.pc), " pc[0]= ", self.pc[0])
        # return self.pc, self.pc_rgb
        return self.transformed_pc, self.transformed_pc_rgb

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

    def evaluate_grasp(self, grasp, iterator = 0, manual=False):
        eval_grasp_action = {}
        eval_grasp_action['EVA_POSE'] = None
        eval_grasp_action['EVA_Z'] = None
        eval_grasp_action['EVA_SUCCESS'], eval_grasp_action['EVA_CLOSURE'] = self.widowx.eval_grasp(manual=manual)
        if eval_grasp_action['EVA_SUCCESS']:
          print("successful grasp: ",  eval_grasp_action['EVA_CLOSURE'])
          eval_grasp_action['EVA_ACTION'] = "RANDOM_DROP"
        elif iterator < 5:
          eval_grasp_action['EVA_ACTION'] = "TRY_AGAIN"
          print("unsuccessful grasp: ", eval_grasp_action['EVA_CLOSURE'])
        else:
          eval_grasp_action['EVA_ACTION'] = "NO_DROP"
          print("unsuccessful grasp2: ", eval_grasp_action['EVA_CLOSURE'])
        # compare keypoints for planned object
        # see if attempted grab of object moved associated keypoints or cluster
        # full credit if grabbed/lifted
        # partial credit if moved
        # no credit / retry with different grasp for same object if untouched

        # if not moved, provide attempted grab, current pose, 3d picture to
        # reinforcement learning NN. Bound move.
        return eval_grasp_action

    def evaluate_world(self, manual=False):
        success, closure = self.widowx.eval_grasp(manual=manual)
        # T, distances, i = icp(A, B)   # nxM matrices
        return success

    def find_cluster_centers(self, pc, labels):
        cluster_centers = []
        for cluster in set(labels):
          if cluster != -1:
            running_sum = np.array([0.0, 0.0, 0.0])
            counter = 0

            # for i in len(pc):
            for i in range(pc.shape[0]):
                if labels[i] == cluster:
                    running_sum += pc[i]
                    counter += 1

            center = running_sum / counter
            center[2] -= Z_OFFSET

            # filter below floor or too high 
            if center[2] > Z_MIN:
                center[2] = Z_MIN
                cluster_centers.append(center)
            # elif center[2] < PRELIFT_HEIGHT:
            elif center[2] < Z_MIN - .02:
                for i, label in enumerate(labels):
                    if label == cluster:
                        labels[i] = -1
                continue
            else:
                cluster_centers.append(center)
        return cluster_centers


    def execute_grasp(self, grasp, manual_label=False):
        try:
            def post_move_sample(prefix):
              self.sample[prefix+'_img'] = self.get_rgbd()
              self.sample[prefix+'_pc'] = self.get_pc()
              self.sample[prefix+'_pose'] = self.get_pose()
              self.sample[prefix+'_joints'] = self.widowx.get_joint_values()
              
            x, y, z, theta = grasp
            print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                  % (x, y, z, theta))
            self.sample['attempted_grasp'] = grasp
            post_move_sample('before')

            assert inside_polygon(
                (x, y, z), END_EFFECTOR_BOUNDS), 'Grasp not in bounds'
            assert self.widowx.orient_to_pregrasp(
                x, y), 'Failed to orient to target'
            post_move_sample('orient')

            prelift_z = min(PRELIFT_HEIGHT, (z - GRIPPER_OFFSET - .02))
            print("move to pre-grasp", prelift_z)
            assert self.widowx.move_to_grasp(x, y, prelift_z, theta), \
                'Failed to reach pre-lift pose'
            post_move_sample('prelift', )

            if len(self.base_z) != 0:
              sect = get_sector(x,y)
              z = min(z, self.base_z[sect])
            z -= GRIPPER_OFFSET
            print("move to grasp", z)
            assert self.widowx.move_to_grasp(
                x, y, z, theta), 'Failed to execute grasp'
            post_move_sample('grasp')

            # print("move to vertical", z)
            # # ARD: retry pose to correct wrong z for some orientations
            # reached = self.widowx.move_to_vertical(z)

            print("close gripper")
            self.widowx.close_gripper()
            post_move_sample('post-grasp')
            # rospy.sleep(2)
            reached = self.widowx.move_to_vertical(prelift_z)
            post_move_sample('postlift')
            eval_grasp_action = self.evaluate_grasp(grasp)
            e_i = 1
            while eval_grasp_action['EVA_ACTION'] == "TRY_AGAIN":
              new_pose = eval_grasp_action['EVA_POSE']
              new_z = eval_grasp_action['EVA_Z']
              plan = self.arm_plan(new_pose)
              self.commander.execute(plan, wait=True)
              reached = self.widowx.move_to_vertical(new_z)
              self.widowx.close_gripper()
              post_move_sample('post_grasp'+e_i)
              if (action == "ROTATE"):                     # rotate 20 degrees
                print("rotate")
                self.widowx.wrist_rotate(DEG20_IN_RADIANS)
                self.widowx.open_gripper()
                post_move_sample('rotate'+e_i)

              print("move to vertical")
	      post_move_sample('postlift'+e_i)
              reached = self.widowx.move_to_vertical(prelift_z)
              eval_grasp_action = evaluate_grasp(grasp, e_i)
              e_i += 1

            # GRAB COMPLETE; DO DROP IF SUCCESSFUL
            if (eval_grasp_action['EVA_ACTION'] in ["RANDOM_DROP","NO_DROP","FLIP_DROP"]):
              self.widowx.move_to_neutral()
              post_move_sample('after')
              success = self.evaluate_world()

              def pos_neg():
                if random() >=.5:
                  return -1
                return 1
              action = "RANDOM_DROP"
              if (action == "NO_DROP"):    
                pass
              elif (action == "RANDOM_DROP"):
                x = random() * self.drop_eeb[0] * pos_neg()
                y = random() * self.drop_eeb[1] * pos_neg()
                # z = PRELIFT_HEIGHT
                print("Orient to drop: ", x, y, prelift_z)
                assert self.widowx.orient_to_pregrasp(
                    x, y), 'Failed to orient to target'
                print("Random drop: ", x, y, z)
                if len(self.base_z) != 0:
                  sect = get_sector(x,y)
                  z = min(z, self.base_z[sect])
                z -= GRIPPER_OFFSET
                assert self.widowx.move_to_grasp(
                  x, y, z, theta), 'Failed to random drop'
                reached = self.widowx.move_to_vertical(z)
              elif (action == "FLIP_DROP"):         # rotate 20 degrees
                assert self.widowx.orient_to_pregrasp(
                    x, y), 'Failed to orient to target'
                print("move to pre-grasp")
                prelift_z = min(PRELIFT_HEIGHT, (z - GRIPPER_OFFSET - .02))
                assert self.widowx.move_to_grasp(x, y, prelift_z, theta), \
                    'Failed to reach pre-lift pose'
                # Flips 90 degrees and drops 
                assert self.widowx.flip_and_drop(), 'Failed to flip and drop'
                print("move to grasp")
                assert self.widowx.move_to_grasp(
                    x, y, z, theta), 'Failed to execute grasp'
              if (action != "NO_DROP"):    
                print("open gripper")
                self.widowx.open_gripper()
                rospy.sleep(2)
                success = self.evaluate_world()
              # policy.evaluate_grasp(self.b4pc, self.b4pc_rgb, self.afterpc, self.afterpc_rgb, grasp
              # success = self.evaluate_grasp3(self.b4pc, self.afterpc)
            else:                                  # original code
              self.widowx.move_to_drop(), 'Failed to move to drop'
              # rospy.sleep(2)
              success = self.evaluate_grasp(manual=manual_label)
              self.sample['gripper_closure'] = self.widowx.eval_grasp()[1]

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
            rgba = int(deg * 4294967294 / 2 / 360 + 255)
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
        if FAVOR_KEYPOINT:
          pc2 = KP.add_to_pc(pc2)
        pc2 = np.reshape(pc2, (len(pc2), 4))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        pc2 = point_cloud2.create_cloud(self.pc_header, fields, pc2)
        self.pc_grasps_pub3.publish(pc2)

    def publish_pc(self, grasps = None, chosen_grasp = None):
      # Derived from:
      # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
      for j in range(2):
        if j == 0 and not DISPLAY_PC_DEPTH_MAP:
          continue
        elif j == 0:
          pc = self.transformed_pc
          # pc_rgb = self.transformed_pc_rgb[:,2]
          self.base_z = compute_z_sectors(self.transformed_pc)
          pc_rgb = pc_depth_mapping(self.transformed_pc, self.base_z)
        elif DISPLAY_PC_RGB and j == 1:
          pc = self.full_pc
          pc_rgb = self.full_pc_rgb
        else:
          continue
        pc2 = []
        for i, p in enumerate(pc):
          p2 = [p[0],p[1],p[2],pc_rgb[i]]
          pc2.append(p2)
        if FAVOR_KEYPOINT:
          pc2 = KP.add_to_pc(pc2)
        pc2 = np.reshape(pc2, (len(pc2), 4))
        # print("pc2 len: ", len(pc2))
        # print("pc2_rgb len: ", len(pc2_rgb))
        # print("pc2[0]: ", pc2[0])
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

    def save_sample(self, i=0):
        self.sample['timestamp'] = time.ctime()

        self.sample['D'] = self.camera_info.D
        self.sample['K'] = self.camera_info.K
        self.sample['R'] = self.camera_info.R
        self.sample['P'] = self.camera_info.P

        graspPoint = np.array([self.sample['pose'][0], self.sample['pose'][1],
                               self.sample['pose'][2]])

        predicted = self.calculate_crop(graspPoint)

        self.sample['pixel_point'] = predicted

        before = self.sample['before_img'][:, :, :3].astype(np.uint8)

        with h5py.File(self.datapath + '/' + str(i) + '.hdf5', 'w') as file:
            for key in self.sample:
                file[key] = self.sample[key]

        self.sample = {}

        print('Saved to %s' % self.datapath  + '/'+ str(i) + '.hdf5')
