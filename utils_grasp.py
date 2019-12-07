from config_grasp import *
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import math

def get_sector(x, y):
      pc_b = BASE_PC_BOUNDS
      # 0,0 is approx center of base
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      x_sz = (abs(ul[0]) + abs(br[0])) / SECTOR_SIZE
      y_sz = (abs(ul[1]) + abs(br[1])) / SECTOR_SIZE
      x_sect = int((x + abs(ul[0])) / x_sz)
      # print(ul[1], y, y_sz)
      y_sect = int((y + abs(ul[1])) / y_sz)
      # print(ul[1], y_sect, y, y_sz)
      sect = (x_sect * SECTOR_SIZE + y_sect)
      return sect

def compute_z_sectors(pc):
      base_z = []
      sector_sz = SECTOR_SIZE  # depends on max object size vs. base sz
      for i in range(sector_sz * sector_sz):
        base_z.append(0)
      for i,p in enumerate(pc):
        sect = get_sector(p[0],p[1])
        if base_z[sect] < p[2]:
          base_z[sect] = p[2]
      return base_z

# Python PCL interface for pcl_segment_cluster does not work in 
# some patch releases.  Workaround with python implementation.
def segment_cluster(pc):
      # seg.set_normal_distance_weight(0.1)
      # seg.set_method_type(pcl.SAC_RANSAC)
      # seg.set_max_iterations(100)
      # seg.set_distance_threshold(0.03)
      max_iterations=100
      best_inliers = None
      n_inliers_to_stop = len(pc)
      self.point = np.mean(pc, axis=0)
      # data is an np.array
      # data_adjust = data - mean
      data_adjust = pc - self.point
      matrix = np.cov(data_adjust.T)  # transpose data_adjust
      eigenvalues, self.normal = np.linalg.eig(matrix)
      n_best_inliers = 0
      max_dist = 1e-4  # 1e-4 = 0.0001.
      print_once = True
      for i in range(max_iterations):
          # k_points = sampler.get_sample()
          normal = np.cross(pc[1] - pc[0], pc[2] - pc[0])
          self.point = pc[0]
          if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            if print_once:
              print_once = False
              print("normal: ",normal)
            self.normal = [1,1,1]
          else:
            self.normal = normal / np.linalg.norm(normal)
          vectors = pc - self.point
          all_distances = np.abs(np.dot(vectors, self.normal))
          inliers = all_distances <= max_dist
          n_inliers = np.sum(inliers)
          if n_inliers > n_best_inliers:
              n_best_inliers = n_inliers
              best_inliers = inliers
              if n_best_inliers > n_inliers_to_stop:
                  break
      # print("plane: ", best_inliers)            # true/false array
      # print("len plane: ", len(best_inliers))
      # print("len pc   : ", len(pc))             # same len as above
      # for i in range(len(best_inliers)):
      #   if i % 10 == 0:
      #     print(pc[i])
      return best_inliers


# Function to find distance
def shortest_distance_from_line(x1, y1, a, b, c):
        d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
        return d

def pc_depth_mapping(pc, base_z):
     pc_depth = []
     for i, p in enumerate(pc):
       # rgbd = abs(int(4294967294 * (p[2] - .44) ))
       sect = get_sector(p[0], p[1])
       height = abs(base_z[sect] - p[2])
       if height < MIN_OBJ_HEIGHT:
         # darker blue
         # 4294967040 = FFFFFF00
         rgbd = abs(int(height * 255 * 255 + 102))
       else:
         # brighter green / red
         rgbd = abs(int(height * 255 * 255 * 255 * 255 - 100))
       pc_depth.append(rgbd)
       # pc_depth.append(p1)
       # p1 = [p[0],p[1],p[2],rgbd]
       # if (i < 10):
       #   print(i," p1: ",p1)
       # pc_depth.append(p1)
     pc_depth = np.reshape(pc_depth, (len(pc_depth), 1))
     return pc_depth


# Maps 3d x/y to 2d x/y and compare to keypoints 2d x/y
# http://docs.ros.org/kinetic/api/librealsense2/html/opencv__pointcloud__viewer_8py_source.html
def from_2d_pixel_to_3d_point(points, pc_3d):
        """project 3d vector array to 2d"""
        min_dist = []      # distance from keypoint x/y to pc x/y
        min_v = []
        p_x_j = []
        p_y_j = []
        dummy_v = pc_3d[0]
        for i in range(len(points)):
          min_dist.append(IMG_WIDTH * IMG_HEIGHT)
          min_v.append(dummy_v)
          p_x_j.append(0)
          p_y_j.append(0)
        pc_points = []
        margins = KP_IMG_PC_MAP[0]
        ratios  = KP_IMG_PC_MAP[1]
        for i in range(len(pc_3d)):
          v = pc_3d[i]
          # print("v:  ",v)
          if v[2] != 0:
            p_x = (v[0] / v[2] * IMG_HEIGHT + IMG_WIDTH/2.0) * ratios[0]
            p_y = (v[1] / v[2] * IMG_HEIGHT + IMG_HEIGHT/2.0) * ratios[1]
            # for j,pt in enumerate(points):
            for j in range(len(points)):
              pt = points[j].pt
              x = pt[0] + margins[0] 
              y = pt[1] + margins[1]
              dist = math.sqrt((p_x - int(x))*(p_x - int(x)) + (p_y - int(y))*(p_y - int(y)))
              if min_dist[j] > dist:
                min_dist[j] = dist
                min_v[j] = v
                p_x_j[j] = p_x
                p_y_j[j] = p_y
        for j,kp in enumerate(points):
          pc_points.append(min_v[j])
        # print("pc_points: ", pc_points)
        return pc_points

