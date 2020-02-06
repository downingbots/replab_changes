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
      # orb = cv.ORB(1000,1.2)         # Initiate SIFT detector
      # find the keypoints and descriptors with SIFT or ORB
      if RGB_DEPTH_FROM_PC:
        # already cropped and transformed
        cropped_img = img  
      else:
        cropped_img = self.crop_img(img)

      self.keypoints, self.descriptor = orb.detectAndCompute(cropped_img,None)

      self.bridge = CvBridge()

      top = KP_IMG_CROP_DIM[0]
      top_margin = KP_IMG_MARGIN_DIM[0]
      bottom_margin = KP_IMG_MARGIN_DIM[1]

      if not RGB_DEPTH_FROM_PC:
        # transform camera to 3d mapping
        for kp in self.keypoints:
          x,y = kp.pt
          # print("pt:",pt)
          kp_left_margin = top_margin[0] + (top_margin[0] - bottom_margin[1]) * (y / (top[1] - top[0]))
          kp_right_margin = top_margin[1] + (top_margin[1] - bottom_margin[0]) * (y / (top[1] - top[0]))
          # filter keypoints out of margins
          if x < kp_left_margin or x > kp_right_margin or y < top[0]:
            self.keypoints.remove(kp)
      if DISPLAY_IMG_KEYPOINT:
          print("num keypoints: ",len(self.keypoints))
          self.publish_img(img)
      if DISPLAY_PC_KEYPOINT:
          self.header = None
          self.pc_kp_pub = rospy.Publisher(PC_KP_TOPIC, PointCloud2, queue_size=1)
          self.pc_subscriber = rospy.Subscriber(
                                 POINTCLOUD_TOPIC, PointCloud2, self.update_pc)


    # descriptor match implementations:
    # https://github.com/opencv/opencv/blob/master/modules/features2d/src/matchers.cpp
    def map_to_clusters(self, clusters):
      # image to cluster mapping
      kp_list = self.get_kp()
      # look through known KPs for matching descriptors for clusters
      for kp in kp_list:
        print("ARD: TODO map_to_clusters")
        # look through clusters for matching points

    def deep_copy_kp(self, KP, kp_i):
      # ARD TODO
      # copy the list of keypoints
      # normalize the keypoints
      # keypoints are based upon the pixel values; transform to x/y/z

      # does len(des) == len(kp)

      # deep copy descriptors?  Not sure this is correct
      n = 0
      des = KP.get_descriptors()
      num_desc = len(des)
      len_desc = len(des[0])
      # copy descriptors into continuous bytes
      s = [0]*(num_desc*len_desc)
      for i in range(num_desc):
        for c in range(len_desc):
          s[n] = des[i,c]
          n = n + 1
      # copy byte offset of each descriptor
      new_desc = [0]*len(s)
      for i in range(0,len(s)):
        new_desc[i]=int(s[i])

      # Get descriptors from second y image using the detected points
      # from the x image
      # f, d = orb.compute(im_y, f)
      # direct deep copy of pixel feature locations
      f = KP.get_features()
      centroid = self.cluster['centroid']
      return [cv2.KeyPoint(x = (k.pt[0]-centroid.x), y = (k.pt[1]-centroid.y),
            _size = k.size, _angle = k.angle,
            _response = k.response, _octave = k.octave,
            _class_id = k.class_id) for k in f], new_desc


    def get_kp(self):
      kp_list = [[kp.pt[0], kp.pt[1]] for kp in self.keypoints]
      return kp_list

    def get_features(self):
      return self.keypoints


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
      # cropped_img = self.crop_img(img)
      cropped_img = self.crop_img(img)
      cropped_img = cv.drawKeypoints(cropped_img,self.keypoints,None,color=(0,255,0), flags=0)
      imgmsg = self.bridge.cv2_to_imgmsg(cropped_img)
      if DISPLAY_IMG_KEYPOINT:
        self.kp_publisher.publish(imgmsg)
      return imgmsg
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
#     For each world cluster, generate cluster 2d image
#     Compute and detect KPs for each cluster 2d image
#     Compare full set of new KPs to each world cluster KPs
#     Map new KPs to new clusters  
#     Map new clusters to world clusters
#     compare and combine clusters
#     Future: handle rotation, flipping, rolling
# 

    def compare_kp(self,pc_clusters,KP2):
      # from matplotlib import pyplot as plt

      # find the keypoints and descriptors with ORB
      kp1 = self.get_kp()
      des1 = self.get_descriptor()
      kp2 = KP2.get_kp()
      des2 = KP2.get_descriptor()
      # if des1.empty() or des2.empty():
      #   print("empty descriptor for keypoint")
      #   return None, 0
      # des1.convertTo(des1, CV_32F); 
      # des2.convertTo(des2, CV_32F);

      ## FLANN parameters
      #FLANN_INDEX_KDTREE = 0
      #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      #search_params = dict(checks=50)   # or pass empty dictionary
      #flann = cv2.FlannBasedMatcher(index_params,search_params)
      #matches = flann.knnMatch(des1,des2,k=2)

      # kp1, des1 = orb.detectAndCompute(img1,None)
      # kp2, des2 = orb.detectAndCompute(img2,None)
      # create BFMatcher object
      bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
      # Match descriptors.
      bf_matches = bf.match(des1,des2)
      print("bf_matches",bf_matches, len(kp1), len(kp2))
      # bf_matches = bf.knnMatch(des1,des2, k=2)
      # for i,(m,n) in enumerate(bf_matches):
      #   bf_ratio = m.distance / n.distance:

      # Initialize lists
      list_kp1 = []
      list_kp2 = []
      list_score = []
      list_ratio = []
      list_cluster = []
      score = 0
      score2 = 0
      for i,m in enumerate(bf_matches):
        if m.distance < 0.70:
      # for i,(m,n) in enumerate(bf_matches):
      #   if m.distance < 0.75*n.distance:
      #   for i,(m,n) in enumerate(matches):
        #   if m.distance < 0.70*n.distance:
          # good.append([kp1])
          # for BF (in cv2.DMatch objects):
          # queryIdx - row of the kp1 interest point matrix that matches
          # trainIdx - row of the kp2 interest point matrix that matches
          # Get the matching keypoints for each of the images
          img1_idx = m.queryIdx
          img2_idx = m.trainIdx
          # x - columns
          # y - rows
          (x1,y1) = kp1[img1_idx].pt
          (x2,y2) = kp2[img2_idx].pt
          # Append to each list
          list_kp1.append((x1, y1))
          list_kp2.append((x2, y2))
          list_ratio.append(m.distance / n.distance)
          found = False
          found_c = None
          for i,c in enumerate(pc_clusters):
            if (x2,y2) in c.shape:
              list_cluster.append(c)
              if found_c == None:
                found_c = c
                print("matching pc cluster found",i)
                found = True
              elif found_c == c:
                print("matching pc cluster found",i)
                found = True
              else:
                print("different matching pc cluster found",i)
              break
          if not found:
              print("matching pc cluster not found")
              list_cluster.append(None)
          # we may be able to figure out rotations of world objects

      score = 0
      score2 = 0
      for i in range(len(list_kp1)):
          # score += m.distance
          # score2 += m.distance / n.distance
          score2 += list_ratio[i]
      # score = (100-(score / len(bf_matches)))
      # score2 = score2 / len(bf_matches)
      if len(list_kp1) > 0:
        score2 = score2 / len(list_kp1)
      else:
        return None, 0
      print("RESULT: Signature match with score = {}".format(score2))
      return list_cluster[0],score2
      
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
      # FLANN_INDEX_KDTREE = 0
      # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      # search_params = dict(checks=50)   # or pass empty dictionary
      # flann = cv.FlannBasedMatcher(index_params,search_params)
      # flann_matches = flann.knnMatch(des1,des2,k=2)
      # for i,(m,n) in enumerate(flann_matches):
      #   flann_ratio = n.distance/m.distance
      
      # Need to draw only good matches, so create a mask
      # matchesMask = [[0,0] for i in xrange(len(matches))]
      # ratio test as per Lowe's paper
      #     if m.distance < 0.7*n.distance:
      #         matchesMask[i]=[1,0]
      # draw_params = dict(matchColor = (0,255,0),
      #                    singlePointColor = (255,0,0),
      #                    matchesMask = matchesMask,
      #                    flags = 0)
      # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
      # plt.imshow(img3,),plt.show()
      # return bf_ratio, flann_ratio 
