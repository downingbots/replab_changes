#!/usr/bin/env python

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import (
    Image,
    PointCloud2,
    CameraInfo
)
from std_msgs.msg import (
    UInt16,
    Int64,
    Float32,
    Header,
    String
)
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError

import argparse
import traceback

import sensor_msgs.msg

from controller import WidowX
from config import *
from utils import *
from utils_grasp import *


class Click2Control:

    def __init__(self, execute=False, calibrate=False):
        # WidowX controller interface
        self.widowx = WidowX()

        # Register subscribers
        self.pc_subscriber = rospy.Subscriber(
            POINTCLOUD_TOPIC, PointCloud2, self.update_pc)
        self.camerainfo_subscriber = rospy.Subscriber(
            DEPTH_CAMERA_INFO_TOPIC, CameraInfo, self.save_caminfo)
        # self.pc_subscriber = rospy.Subscriber(
        #     "/camera/depth/image_rect_raw", PointCloud2, self.update_pc)
        # self.action_subscriber = rospy.Subscriber(
        #     RGB_IMAGE_TOPIC, Image, self.update_rgb)
, pc2)
        self.pc_rgb_pub1
        if DISPLAY_PC_RGB:
          self.pc_rgb_pub1 = rospy.Publisher(PC_RGB_TOPIC, PointCloud2, queue_size=1)
        # overloads PC_GRASPS topic to publish selected points 
        if DISPLAY_PC_GRASPS:
          self.pc_points_pub2 = rospy.Publisher(PC_GRASPS_TOPIC, PointCloud2, queue_size=1)
        # Store latest RGB-D
        self.rgb = None
        self.depth = None
        # self.pc1 = None
        self.pc_ready = True
        self.pc = []

        self.bridge = CvBridge()

        self.cm = CALIBRATION_MATRIX

        self.robot_coordinates = []
        self.camera_coordinates = []
        self.camera_info = sensor_msgs.msg.CameraInfo()

        self.execute = execute
        self.calibrate = calibrate

        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(self.camera_info)

    def update_rgb(self, data):
        self.rgb = self.bridge.imgmsg_to_cv2(data)

    def update_d(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data)

    def update_pc(self, data):
        # print("update_pc")
        # not enough data? only 261322 +/- 1000 of 307200 processed
        # real points, not padded
        # self.pc = list(pc2.read_points(data, skip_nans=False))
        # import copy 
        if not self.pc_ready:
          # busy materializing from previous pointcloud.
          # skip and process next point cloud instead
          print("skip update_pc")
          return
        self.pc_header = data.header
        self.pc_ready = False
        self.pc = list(pc2.read_points(data, skip_nans=False))
        self.pc_ready = True
        # self.pc = pc2.read_points(data, skip_nans=True)


        # if self.pc_ready == 1:
          # self.pc = []
          # for w in range(640):
            # for h in range(480):
              # generators are lazily evaluated; put in list, then find out
              # this code doesn't work either.
              # not enough data? only 261322 +/- 1000 of 307200 processed
              # pc_point = (pc2.read_points(data, skip_nans=True, uvs=[w, h]))
              # self.pc.append( copy.deepcopy( pc_point))
              # print(w," ", h)
              # self.pc.append(list(pc2.read_points(data, skip_nans=False, uvs=[[w, h]])))
          # print("update_pc ", len(self.pc))
          # self.pc_ready = 0
        # else:
          # self.pc1 = []
          # for w in range(640):
            # for h in range(480):
              # pc_point = (pc2.read_points(data, skip_nans=True, uvs=[w, h]))
              # self.pc1.append( copy.deepcopy( pc_point))
              # print(w," ", h)
              # self.pc1.append(list(pc2.read_points(data, skip_nans=False, uvs=[[w, h]])))
          # print("update_pc ", len(self.pc1))
          # self.pc_ready = 1
        # self.pc = list(pc2.read_points(data, skip_nans=False))
        # self.pc = list(data)

    def save_caminfo(self, data):
        self.cam_info = data


    def take_action(self, data):
        print("take_action")
        x_2d, y_2d = data.data.split('(')[1].split(')')[0].split(',')
        # x,y are 2d rgb image pixel locations.
        # need to translate to 3-d pointcloud location
        # see https://docs.ros.org/api/image_geometry/html/python/
        # see https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0
        # projectPixelTo3dRay((u,v))
        # pt3d = self.camera.projectPixelTo3dRay((x,y))
        # disparities = lcorners[:,:,0] - rcorners[:,:,0]
        # d = disparities[i,0]
        # getDisparity(Z)
        # for sterio cameras:
        # d = self.camera.getDisparity(Z_OFFSET)
        # pc_point = self.camera.projectPixelTo3d((x,y), d)
        # pc_point = self.camera.projectPixelTo3d((x,y))
        # pc_point = rs2_project_point_to_pixel(intrinsics, [x, x, x])

        # register_index = int(x) + (int(y) * 640)
        while not self.pc_ready:
          pass
        self.pc_ready = False
        pc_3d = self.pc
        # pc_3d = np.array(self.pc)
        self.pc_ready = True

        def transform_pc(srcpc, tf_matrix):
            ones = np.ones((srcpc.shape[0], 1))
            srcpc = np.append(srcpc, ones, axis=1)
            out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc_3d = transform_pc(pc_3d, self.cm)

        pnt_2d = [[x_2d, y_2d]]
        pc_point = from_2d_pixel_to_3d_point(pnt_2d,pc_3d) 

        # if self.pc_ready == 0:
          # pc_point = self.pc[register_index]
        # else:
          # pc_point = self.pc1[register_index]

        print('Pixel: (%s, %s)' % (x_2d, y_2d))
        print('PC point: ' + str(pc_point))

        if pc_point[2] == 0:
            print('No depth reading')
            print(pc_depth)
            return

        if self.execute:
            pc_point = np.array([pc_point[0], pc_point[1], pc_point[2], 1.])
            transformed = np.dot(self.cm, pc_point)[:3]
            print(transformed)

            theta = 0.0
            grasp = np.concatenate([transformed, [theta]], axis=0)
            grasp[2] -= Z_OFFSET
            # grasp = np.array([x, y, Z_OFFSET, theta])

            print('Grasp: ' + str(grasp))
            self.execute_grasp(grasp)
            self.widowx.open_gripper()
            self.widowx.move_to_neutral()

        elif self.calibrate:
            user_in = raw_input('Keep recorded point? [y/n] ')

            if user_in == 'y':
                pose = self.widowx.get_current_pose().pose.position
                print(pose.x, pose.y, pose.z)

                self.robot_coordinates.append((pose.x, pose.y, pose.z))
                self.camera_coordinates.append(pc_point[:3])
                print('Saved')

            elif user_in == 'n':
                print('Not saved')
        if DISPLAY_PC_RGB:
          self.publish_pc(pc_3d):
        if DISPLAY_PC_GRASPS:
          self.publish_points(pc_point):

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

    def execute_grasp(self, grasp):
        try:
            x, y, z, theta = grasp

            print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                  % (x, y, z, theta))

            assert inside_polygon(
                (x, y, z), END_EFFECTOR_BOUNDS), 'Grasp not in bounds'

            assert self.widowx.orient_to_pregrasp(
                x, y), 'Failed to orient to target'

            assert self.widowx.move_to_grasp(x, y, PRELIFT_HEIGHT, theta), \
                'Failed to reach pre-lift pose'

            assert self.widowx.move_to_grasp(
                x, y, z, theta), 'Failed to execute grasp'

            self.widowx.close_gripper()

            reached = self.widowx.move_to_vertical(PRELIFT_HEIGHT)

            assert self.widowx.move_to_drop(), 'Failed to move to drop'

        except Exception as e:
            print('Error executing grasp -- returning...')
            traceback.print_exc(e)


    def publish_points(self, 3d_points):
        pc2 = []
        if grasps is not None:
          for i, (g) in enumerate(3d_points):
                # bright yellow =  rgb(255,255,0) transparency = 0
                WHITE = 4294967294 / 2
                rgba = WHITE
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
        self.pc_points_pub2.publish(pc2)

    def publish_pc(self, pc):
        pc2 = []
        for i, p in enumerate(pc):
          p2 = [p[0],p[1],p[2],pc2_rgb[i]]
          pc2.append(p2)
        pc2 = np.reshape(pc2, (len(pc2), 4))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1)]
                  PointField('rgb', 12, PointField.UINT32, 1)]
        pc2 = point_cloud2.create_cloud(self.pc_header, fields, pc2)
        self.pc_rgb_pub1.publish(pc2)

def main():
    parser = argparse.ArgumentParser(
        description="Executes user-specified grasps from a GUI window")
    parser.add_argument('--debug', action="store_true", default=False,
                        help="Prevents grasp from being executed (for debugging purposes)")
    args = parser.parse_args()

    rospy.init_node("click2control")

    executor = Click2Control(execute=(not args.debug), calibrate=False)
    print('Run commander_human.py to issue commands from the GUI window')

    executor.widowx.move_to_neutral()

    rospy.sleep(2)
    rospy.spin()

if __name__ == '__main__':
    main()
