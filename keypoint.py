#!/usr/bin/env python
# from replab_grasping.utils_grasping import *
import rospy
from sensor_msgs.msg import (Image)
from replab_core.config import *

from utils_grasp import *
from config_grasp import *
from cv_bridge import CvBridge
import utils_grasp
import cv2 as cv

class Keypoints:
    def crop_img(self, img):
      top = KP_IMG_CROP_DIM[0]
      bottom = KP_IMG_CROP_DIM[1]
      crop_img = img[top[0]:top[1], bottom[0]:bottom[1]]
      return crop_img

    def __init__(self, img):
      if DISPLAY_IMG_KEYPOINT:
        self.kp_publisher = rospy.Publisher(IMG_KEYPOINT_TOPIC, Image, queue_size=1)
      self.keypoints = []
      self.pc_header = None
      orb = cv.ORB_create()         # Initiate SIFT detector
      # find the keypoints and descriptors with SIFT
      if RGB_DEPTH_FROM_PC:
        # already cropped and trasnformed
        cropped_img = img  
      else:
        cropped_img = self.crop_img(img)
      self.keypoints, self.descriptor = orb.detectAndCompute(cropped_img,None)

      self.bridge = CvBridge()

      top = KP_IMG_CROP_DIM[0]
      top_margin = KP_IMG_MARGIN_DIM[0]
      bottom_margin = KP_IMG_MARGIN_DIM[1]

      if !RGB_DEPTH_FROM_PC:
        # transform camera to 3d mapping
        for kp in self.keypoints:
          x,y = kp.pt
          # print("pt:",pt)
          kp_left_margin = top_margin[0] + (top_margin[0] - bottom_margin[1]) * (y / (top[1] - top[0]))
          kp_right_margin = top_margin[1] + (top_margin[1] - bottom_margin[0]) * (y / (top[1] - top[0]))
          # filter keypoints out of margins
          if x < kp_left_margin or x > kp_right_margin or y < top[0]:
            self.keypoints.remove(kp)
      # print("num keypoints: ",len(self.keypoints))
      if DISPLAY_PC_KEYPOINT:
          self.header = None
          self.pc_kp_pub = rospy.Publisher(PC_KP_TOPIC, PointCloud2, queue_size=1)
          self.pc_subscriber = rospy.Subscriber(
                                 POINTCLOUD_TOPIC, PointCloud2, self.update_pc)


    def get_kp(self):
      kp_list = [[kp.pt[0], kp.pt[1]] for kp in self.keypoints]
      return kp_list

    def get_descriptor(self):
      return self.descriptor

    # Maps 3d x/y to 2d x/y and compare to keypoints 2d x/y
    # http://docs.ros.org/kinetic/api/librealsense2/html/opencv__pointcloud__viewer_8py_source.html
    def kp_to_3d_point(self, pc_3d):
      return from_2d_pixel_to_3d_point(self.keypoints, pc_3d)

    def update_pc(self, data):
      self.pc_header = data.header

    def publish_pc(self, pc):
      if not DISPLAY_PC_KEYPOINT or self.pc_header == None:
        return
      # for 3d display
      # kp = from_2d_pixel_to_3d_point(self.get_kp(), pc)
      kp = from_2d_pixel_to_3d_point(self.keypoints, pc)
      color = 4294967294 / 8
      kp_pc = [[p[0],p[1],p[2],color] for p_i, p in enumerate(kp)]
      kp_pc = np.reshape(kp_pc, (len(kp_pc), 4))
      fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # PointField('rgba', 12, PointField.UINT32, 1)]
                PointField('rgb', 12, PointField.UINT32, 1)]
      kp_pc = point_cloud2.create_cloud(self.pc_header, fields, kp_pc)
      self.pc_kp_pub.publish(kp_pc)

    def publish_img(self, img):
      cropped_img = self.crop_img(img)
      cropped_img = cv.drawKeypoints(cropped_img,self.keypoints,None,color=(0,255,0), flags=0)
      imgmsg = self.bridge.cv2_to_imgmsg(cropped_img)
      if DISPLAY_IMG_KEYPOINT:
        self.kp_publisher.publish(imgmsg)
      return imgmsg

    def compare_kp(KP2):
      # from matplotlib import pyplot as plt

      # find the keypoints and descriptors with SIFT
      # kp1, des1 = orb.detectAndCompute(img1,None)
      # kp2, des2 = orb.detectAndCompute(img2,None)
      # create BFMatcher object
      bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
      # Match descriptors.
      bf_matches = bf.match(des1,des2)
      
      # Sort them in the order of their distance.
      # matches = sorted(bf_matches, key = lambda x:x.distance)
      # Draw first 10 matches.
      # img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
      # plt.imshow(img3),plt.show()
      ####
      # BFMatcher with default params
      # bf = cv.BFMatcher()
      # bf_matches = bf.knnMatch(des1,des2, k=2)
      # Apply ratio test
      # good = []
      # for m,n in matches:
      #     if m.distance < 0.75*n.distance:
      #         good.append([m])
      # cv.drawMatchesKnn expects list of lists as matches.
      # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
      # plt.imshow(img3),plt.show()
      ####
      # FLANN parameters
      FLANN_INDEX_KDTREE = 0
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)   # or pass empty dictionary
      
      flann = cv.FlannBasedMatcher(index_params,search_params)
      flann_matches = flann.knnMatch(des1,des2,k=2)
      
      # Need to draw only good matches, so create a mask
      # matchesMask = [[0,0] for i in xrange(len(matches))]
      # ratio test as per Lowe's paper
      # for i,(m,n) in enumerate(matches):
      #     if m.distance < 0.7*n.distance:
      #         matchesMask[i]=[1,0]
      # draw_params = dict(matchColor = (0,255,0),
      #                    singlePointColor = (255,0,0),
      #                    matchesMask = matchesMask,
      #                    flags = 0)
      # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
      # plt.imshow(img3,),plt.show()

      return bf_matches, flann_matches 
