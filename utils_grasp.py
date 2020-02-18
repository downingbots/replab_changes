from config_grasp import *
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter
from replab_core.utils import *

# Current values:
# +-.17 by +-.15 tray
# 136 x 120 produces 16320 ~.1 inches pixels
def rgb_depth_map_from_pc(pc, rgb_pc, fill=True, init_blank=False):
      # rgb = np.zeros((RGB_WIDTH, RGB_HEIGHT, 1))
      rgb = np.zeros( (RGB_WIDTH,RGB_HEIGHT, 3), dtype=np.uint8)
      depth = np.zeros((RGB_WIDTH, RGB_HEIGHT, 1))
      pc_map = np.full((RGB_WIDTH, RGB_HEIGHT), -1)
      # pc_img = np.zeros((RGB_WIDTH, RGB_HEIGHT, 4))
      pc_img = []
      pc_b = BASE_PC_BOUNDS
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      # print("ul ",ul," br ",br)
      x_sz = (abs(ul[0]) + abs(br[0])) 
      y_sz = (abs(ul[1]) + abs(br[1])) 
      # print("x_sz ",x_sz," y_sz ",y_sz)

      # min_p0 = 100000000
      # max_p0 = -100000000
      # min_p1 = 100000000
      # max_p1 = -100000000
      # for i, p in enumerate(pc):
        # min_p0 = min(p[0],min_p0)
        # max_p0 = max(p[0],max_p0) 
        # min_p1 = min(p[1],min_p1) 
        # max_p1 = max(p[1],max_p1) 
      # sz_p0 = max_p0 - min_p0 
      # sz_p1 = max_p1 - min_p1 
      # print("min/max", min_p0,max_p0,min_p1,max_p1)
      # random pc should have more pc values than RGB_WIDTH/RGB_HEIGHT
      # convert to fixed RGB_WIDTH/RGB_HEIGHT
      for i, p in enumerate(pc):
        # inside_polygon works for before transform and filter already applied
        # if not inside_polygon(p, BASE_PC_BOUNDS, BASE_HEIGHT_BOUNDS):
        #   continue
        # 0,0 is approx center of base before conversion to x/y
        # x = int((RGB_WIDTH-1)  * (p[0] + (x_sz / 2)) / x_sz +.5) 
        # y = int((RGB_HEIGHT-1) * (p[1] + (y_sz / 2)) / y_sz +.5)
        # x = int((RGB_WIDTH-1)  * (p[0] - min_p0) / sz_p0 + .5)
        # y = int((RGB_HEIGHT-1)  * (p[1] - min_p1) / sz_p1 + .5)
        x = int((RGB_WIDTH-1)  * max(min((p[0] + (x_sz / 2)),x_sz),0) / x_sz +.5) 
        y = int((RGB_HEIGHT-1)  * max(min((p[1] + (y_sz / 2)),y_sz),0) / y_sz +.5) 
        # print("p0 ", p[0], " p1 ", p[1], " x ", x, " y ", y)
        if pc_map[x,y] == -1:
          # print(i, " pc_map", x,y,p)
          # rgb[x,y] = p[3]
          rgb[x,y] = rgb_pc[i][3]
          depth[x,y] = p[2]
          pc_map[x,y] = i
        else:
          # take the highest point for img as we're looking at 3d shape from top
          if (depth[x,y] > p[2]):
            # rgb[x,y] = p[3]
            rgb[x,y] = rgb_pc[i][3]
            depth[x,y] = p[2]
            pc_map[x,y] = i
            # print(i, " rep ",x,y)
          # else:
            # print(i, " dup ",x,y)
          ## take the pt closest to center of pixel
          ## then direct map to PC pixel (p[0],p[1],p[2]), which may 
          ## be important for KPs
          ## alternate approach: average multiple pc values 
          #pp = pc[pc_map[x,y]]
          #centerx0 = (RGB_WIDTH  * (pp[0] + (x_sz / 2))) % x_sz
          #centery0 = (RGB_WIDTH  * (pp[1] + (y_sz / 2))) % y_sz
          #dist0 = sqrt((.5 - centerx0)**2 + (.5 - centery)**2)
          #centerx1 = (RGB_WIDTH  * (p[0] + (x_sz / 2))) % x_sz
          #centery1 = (RGB_WIDTH  * (p[1] + (y_sz / 2))) % y_sz
          #dist1 = sqrt((.5 - centerx1)**2 + (.5 - center1)**2)
          #if (dist1 < dist0):
          #  rgb[x,y] = p[3]
          #  depth[x,y] = p[2]
          #  pc_map[x,y] = i
      # handle random PC points not covering all x,y points in rgb
      # ARD: vectorize
      if fill:  # move inside loop so that pc_img can still be computed
        c = 0
        c2 = 0
        for x in range(RGB_WIDTH):
          for y in range(RGB_HEIGHT):
            if pc_map[x,y] == -1:
              # if x == int(RGB_WIDTH/2):
                # print("fill ",x,y)
              s = 0
              d = 0
              n = 0
              i = []
              for delta1 in (-1,0,1):
                for delta2 in (-1,0,1):
                  if (0 <= (x + delta1) < RGB_WIDTH and
                      0 <= (y + delta2) < RGB_HEIGHT and
                      pc_map[x+delta1,y+delta2] != -1):
                    # s += rgb[x+delta1,y+delta2]
                    # d += depth[x+delta1,y+delta2]
                    # n += 1
                    rgb[x,y] = rgb[x+delta1,y+delta2]
                    pc_map[x,y] = pc_map[x+delta1,y+delta2]
             
              # print("s n d ",s,n,d, " x y ", x,y)
              if n > 0:
                pass
                # rgb[x,y] = int(s / n)
                # depth[x,y] = d / n
              else:
                rgb[x,y] = 0
                depth[x,y] = 0
            if pc_map[x,y] != -1:
              p = rgb_pc[pc_map[x,y]]
              # if x == 0 and y < 10:
                # print(" -> ",pc_map[x,y],p)
              pc_img.append(p)
              c2 += 1
            else:
              c += 1
      else: # no fill, for Keypoints on a single cluster
        for x in range(RGB_WIDTH):
          for y in range(RGB_HEIGHT):
            if pc_map[x,y] != -1:
              pc_img.append(rgb_pc[pc_map[x,y]])
            else:
              pc_img.append(0)    # black background

      # print("pc_map: ",c2, " nomap ",c)
      return rgb, depth, pc_map, pc_img

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

def distance_2d(pt1, pt2):
   return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def distance_3d(pt1, pt2):
   return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

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
        # print("dummy_v", dummy_v)
        for i in range(len(points)):
          min_dist.append(IMG_WIDTH * IMG_HEIGHT)
          min_v.append(dummy_v)
          p_x_j.append(0)
          p_y_j.append(0)
        pc_points = []
        if RGB_DEPTH_FROM_PC:
          pc_b = BASE_PC_BOUNDS
          ul = pc_b[0]  # upper left
          br = pc_b[2]  # bottom right
        else:
          margins = KP_IMG_PC_MAP[0]
          ratios  = KP_IMG_PC_MAP[1]
        for i in range(len(pc_3d)):
          v = pc_3d[i]
          # print("v:  ",v)
          if v[2] != 0:
            if RGB_DEPTH_FROM_PC:
              p_x = (v[0] / v[2] * RGB_HEIGHT + RGB_WIDTH/2.0) 
              p_y = (v[1] / v[2] * RGB_HEIGHT + RGB_HEIGHT/2.0) 
            else:
              p_x = (v[0] / v[2] * IMG_HEIGHT + IMG_WIDTH/2.0) * ratios[0]
              p_y = (v[1] / v[2] * IMG_HEIGHT + IMG_HEIGHT/2.0) * ratios[1]
            # for j,pt in enumerate(points):
            for j in range(len(points)):
              pt = points[j].pt
              if RGB_DEPTH_FROM_PC:
                # x = pt[0] * (abs(ul[0]) + abs(br[0])) / RGB_WIDTH
                # y = pt[1] * (abs(ul[1]) + abs(br[1])) / RGB_HEIGHT
                x = pt[0]
                y = pt[1]
              else:
                x = pt[0] + margins[0] 
                y = pt[1] + margins[1]
              dist = math.sqrt((p_x - int(x))*(p_x - int(x)) + (p_y - int(y))*(p_y - int(y)))
              if min_dist[j] > dist:
                min_dist[j] = dist
                min_v[j] = v
                # print("dist,v:", dist, v, pt, p_x, p_y, x, y)
                p_x_j[j] = p_x
                p_y_j[j] = p_y
        for j,kp in enumerate(points):
          pc_points.append(min_v[j])
        # print("pc_points: ", pc_points)
        return pc_points


