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

def pt_lst_eq(p1, p2):
      return (p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2])

def pt_in_lst(p1, lst):
      return any(pt_lst_eq(p1,p2) for p2 in lst)

def check_bounds(x,y):
      pc_b = BASE_PC_BOUNDS
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      if x <= ul[0]:
        x = ul[0] + OCTOMAP_RESOLUTION
      elif x >= br[0]:
        x = br[0] - OCTOMAP_RESOLUTION
      if y >= ul[1]:
        x = ul[1] - OCTOMAP_RESOLUTION
      elif y <= br[1]:
        x = br[0] + OCTOMAP_RESOLUTION
      return x,y

def close_to_side(pt):
      def compute_x_y(i, j):
            # j for top/bot
            # i for l/r
            # ul = pc_b[0]  # upper left
            # br = pc_b[2]  # bottom right
            pc_b = BASE_PC_BOUNDS
            desired_dist = 1 * INCH 
            p_x = pc_b[i*2][0]
            if (p_x < 0):
              p_x1 = p_x + GRIPPER_WIDTH / 2
              p_x2 = p_x + desired_dist*2 + GRIPPER_WIDTH / 2
            else:
              p_x1 = p_x - GRIPPER_WIDTH / 2
              p_x2 = p_x - desired_dist*2 - GRIPPER_WIDTH / 2
            p_y1 = pc_b[j*2][1]
            if (p_y1 < 0):
              p_y0 = p_y1 - desired_dist
              p_y2 = p_y1 + desired_dist*2
            else:
              p_y0 = p_y1 + desired_dist
              p_y2 = p_y1 - desired_dist*2
            return p_x, p_x1, p_x2, p_y0, p_y1, p_y2

      desired_dist = 1 * INCH 
      pc_b = BASE_PC_BOUNDS
      # 0,0 is approx center of base
      # BASE_PC_BOUNDS = [(-.17,  .15),  top right
      # 		  ( .17,  .15),  top left
      # 		  ( .17, -.15),  bottom left
      # 		  (-.17, -.15)]  bottom right
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      maxval = max(abs(pc_b[0][0])*2, abs(pc_b[0][1]*2))
      min_d = []
      for i in range(2):    # from left-right or top-bottom
        for j in range(2):  # from upper-left or bottom-right
           p_x, p_x1, p_x2, p_y0, p_y1, p_y2 = compute_x_y(i, j)

           # distance to corner
           d_corner = distance_2d([p_x, p_y1], pt)
           d_x      = abs(p_x - pt[0])
           d_y      = abs(p_y1 - pt[1])

           # twice d_x is same, twice d_y is same.
           # at most, once both d_x and d_y are same at same time.
           if ((d_x < desired_dist and d_x < maxval) and
               (d_y < desired_dist and d_y < maxval)):
             min_d = [i,j,d_x, d_y]
             break
           elif (d_x < desired_dist and d_x < maxval):
             min_d = [i,j,d_x, None]
           elif (d_y < desired_dist and d_y < maxval):
             min_d = [i,j,None, d_y]
      if len(min_d) == 1:
        [i,j,d_x, d_y] = min_d
      if len(min_d) == 0:
        # return pt, "no side", 2*desired_dist, None, None, None
        return None
      elif len(min_d) == 1 and (d_x == None or d_y == None):
        p_x, p_x1, p_x2, p_y0, p_y1, p_y2 = compute_x_y(i, j)
        if j == 0:    # left/right side
          if i == 0:
            side = "top right"
            spnt0 = None
            spnt1 = [p_x1, pt[1-i], pt[2]]
            spnt2 = [p_x2, pt[1-i], pt[2]]
          else:
            side = "top left"
            spnt0 = None
            spnt1 = [p_x1, pt[1-i], pt[2]]
            spnt2 = [p_x2, pt[1-i], pt[2]]
        else:         # top/bottom
          spnt1 = [pt[i], p_y1, pt[2]]
          if i == 0:
            side = "bottom right"
            spnt0 = [pt[i], p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt2 = [pt[i], p_y2, pt[2]]
          else:
            side = "bottom left"
            spnt0 = [pt[i], p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt2 = [pt[i], p_y2, pt[2]]
        return pt, side, d1, d2, spnt0, spnt1, spnt2
      elif len(min_d) == 1 and (d_x != None and d_y != None):
        [i,j,d1, d2] = min_d[0]
        p_x, p_x1, p_x2, p_y0, p_y1, p_y2 = compute_x_y(i, j)
        if j == 0:    
          if i == 0: # top, right
            side = "corner: top right"
            spnt0 = [p_x, p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt1 = [p_x1, p_y1, pt[2]]
            spnt2 = [p_x2, p_y2, pt[2]]
          else:      # bottom, right
            side = "corner: bottom right"
            spnt0 = [p_x, p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt1 = [p_x1, p_y1, pt[2]]
            spnt2 = [p_x2, p_y2, pt[2]]
        else:         
          if i == 0: # top, left
            side = "corner: top left"
            spnt0 = [p_x, p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt1 = [p_x1, p_y1, pt[2]]
            spnt2 = [p_x2, p_y2, pt[2]]
          else:      # bottom, left
            side = "corner: bottom left"
            spnt0 = [p_x, p_y0, pt[2] - desired_dist - .5*GRIPPER_LEN]
            spnt1 = [p_x1, p_y1, pt[2]]
            spnt2 = [p_x2, p_y2, pt[2]]
        # pt = check if point is close to side
        # spnt0 = up slope point on top/bot side, None for l/r sides (start point)
        # spnt1 = point on side (middle point)
        # spnt2 = point a minimum distance from side (destination point)
        return pt, side, d1, d2, spnt0, spnt1, spnt2
      else:
        print("ERROR: close to",len(min_d)," sides! ", pt)
        print("min_d: ", min_d)
        # return pt, None, None, None, None, None
        return None

def get_sector(x, y):
      pc_b = BASE_PC_BOUNDS
      # 0,0 is approx center of base
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      x_sz = (abs(ul[0]) + abs(br[0])) / SECTOR_SIZE
      y_sz = (abs(ul[1]) + abs(br[1])) / SECTOR_SIZE
      x_sect = min((SECTOR_SIZE-1),max(0,int((x + abs(ul[0])) / x_sz)))
      # print(ul[1], y, y_sz)
      y_sect = min((SECTOR_SIZE-1),max(0,int((y + abs(ul[1])) / y_sz)))
      # print(ul[1], y_sect, y, y_sz)
      sect = (x_sect * SECTOR_SIZE + y_sect)
      # print(x_sect, y_sect, sect)
      return sect

def compute_z_sectors_from_base(base_pc):
      sector_sz = SECTOR_SIZE  # depends on max object size vs. base sz
      base_z = []
      min_sect = BIGNUM
      for i in range(sector_sz * sector_sz):
        base_z.append(BIGNUM)
      for i,p in enumerate(base_pc):
        sect = get_sector(p[0],p[1])
        if base_z[sect] > p[2]:
          base_z[sect] = p[2]
          if min_sect > p[2]:
            base_z[sect] = p[2]
        if base_z[sect] == BIGNUM:
          print(sect, p)
      for i in range(SECTOR_SIZE*SECTOR_SIZE):
        if base_z[i] == BIGNUM:
          # force all black
          base_z[i] = min_sect
      print("base_z: ", base_z )
      # print("min: ", min_x, min_y)
      return base_z

def compute_z_sectors(pc, prev_base_z = None):
      sector_sz = SECTOR_SIZE  # depends on max object size vs. base sz
      if prev_base_z != None:
        base_z = prev_base_z
      else:
        base_z = []
        for i in range(sector_sz * sector_sz):
          base_z.append(0)
      # min_y = BIGNUM
      # min_x = BIGNUM
      for i,p in enumerate(pc):
        sect = get_sector(p[0],p[1])
        # print("p:", p)
        # print("base_z:", base_z[sect])
        if base_z[sect] < p[2]:
          base_z[sect] = p[2]
        if base_z[sect] == 0:
          print(sect, p)
        # min_y = min(min_y, p[1])
        # min_x = min(min_x, p[0])
      # bz2 = []
      # for i in range(SECTOR_SIZE*SECTOR_SIZE):
      #   bz2.append(base_z[i] - MIN_OBJ_HEIGHT)
      for i in range(SECTOR_SIZE*SECTOR_SIZE):
        if base_z[i] == 0:
          # force all black
          base_z[i] = .55 
      print("base_z: ", base_z )
      # print("min: ", min_x, min_y)
      return base_z

# From:
# https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
def getChar():
      try:
        # for POSIX-based systems (with termios & tty support)
        import tty, sys, termios  # raises ImportError if unsupported
        fd = sys.stdin.fileno()
        oldSettings = termios.tcgetattr(fd) 
        try:
            tty.setcbreak(fd)
            answer = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, oldSettings)
        return answer

      except ImportError:
        # for Windows-based systems
        import msvcrt # If successful, we are on Windows
        return msvcrt.getch()

def distance_2d(pt1, pt2):
   return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def distance_3d(pt1, pt2):
   return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)


def get_pc_min_max(cluster_pc):
        x,y,z = 0,1,2
        # # print("len cluster_pc", len(cluster_pc))
        min_x = BIGNUM
        max_x = 0
        min_y = BIGNUM
        max_y = 0
        min_z = BIGNUM
        max_z = 0
        for pnt_id, pnt in enumerate(cluster_pc):
          min_x = min(min_x, pnt[x]) 
          max_x = max(max_x, pnt[x]) 
          min_y = min(min_y, pnt[y]) 
          max_y = max(max_y, pnt[y]) 
          min_z = min(min_z, pnt[z]) 
          max_z = max(max_z, pnt[z]) 
        # print("Min/max x:",min_x, max_x)
        # print("Min/max y:",min_y, max_y)
        # print("Min/max z:",min_z, max_z)
        return [[min_x, min_y, min_z], [max_x, max_y, max_z]]

def get_approx_octmap_density(numpts):
      pc_b = BASE_PC_BOUNDS
      # 0,0 is approx center of base
      ul = pc_b[0]  # upper left
      br = pc_b[2]  # bottom right
      x_sz = (abs(ul[0]) + abs(br[0])) 
      y_sz = (abs(ul[1]) + abs(br[1])) 
      res = OCTOMAP_RESOLUTION
      exp_num_pnts = (x_sz * y_sz) / res / res
      dist = res * exp_num_pnts / numpts 
      dist = min(dist, res)
      return dist

def build_clusters(pc):
      from rotor import Rotor

      def take_z_axis(p):
        return -p[2]

      for i,p in enumerate(sort_by_dist):
        distances.append([i, p[2]])
      distances = np.sort(distances, axis=0)
      rotor = Rotor()
      rotor.fit_rotate(distances)
      elbow_index = rotor.get_elbow_index()
      z_filter = []
      for e in range(elbow_index):
        if e < elbow_index-1 and distances[e][1] != distances[e+1][1]:
          z_filter.append(distances[e][1])

      # not_base = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc if p[2] not in z_filter]
      not_base = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc if p[2] in z_filter]
      return not_base



def rm_base(pc):
      from rotor import Rotor

      def take_z_axis(p):
        return p[2]

      distances = []
      sort_by_dist = sorted(pc, key=take_z_axis)
      for i,p in enumerate(sort_by_dist):
        distances.append([i, p[2]])
      distances = np.sort(distances, axis=0)
      rotor = Rotor()
      rotor.fit_rotate(distances)
      elbow_index = rotor.get_elbow_index()
      # z_filter = []
      # for e in range(elbow_index):
      #   if e < elbow_index-1 and distances[e][1] != distances[e+1][1]:
      #     z_filter.append(distances[e][1])
      z_filter = [distances[0][1]]
      print("z_filter: ", z_filter)
      # not_base = [p for p in pc if p[2] not in z_filter]
      # not_base = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc if p[2] not in z_filter]
      not_base = [[p[0],p[1],p[2],np.uint32(p[3])] for p in pc if p[2] in z_filter]
      return not_base
      # return not_base[elbow_index][1]
      # return distances[elbow_index][1]

# Python PCL interface for pcl_segment_cluster does not work in 
# some patch releases.  Workaround with python implementation.
def segment_cluster(pc1):
      # seg.set_normal_distance_weight(0.1)
      # seg.set_method_type(pcl.SAC_RANSAC)
      # seg.set_max_iterations(100)
      # seg.set_distance_threshold(0.03)
      # pc = [[p[0],p[1],p[2]] for p in pc1]
      pc = np.array(pc1)[:, :3]
      max_iterations=100
      best_inliers = None
      n_inliers_to_stop = len(pc)
      point = np.mean(pc, axis=0)
      # data is an np.array
      # data_adjust = data - mean
      data_adjust = pc - point
      matrix = np.cov(data_adjust.T)  # transpose data_adjust
      eigenvalues, normal = np.linalg.eig(matrix)
      n_best_inliers = 0
      # max_dist = 1e-4  # 1e-4 = 0.0001.
      # max_dist = 1e-3  # 1e-4 = 0.0001.
      # max_dist = sqrt(0.000255 * 0.000255 * 2)
      # max_dist = 0.000361
      # max_dist = 0.00785
      # max_dist = 0.009
      max_dist = 0.0006
      print_once = True
      for i in range(max_iterations):
          # k_points = sampler.get_sample()
          normal = np.cross(pc[1] - pc[0], pc[2] - pc[0])
          point = pc[0]
          if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            if print_once:
              print_once = False
              print("normal: ",normal)
            normal = [1,1,1]
          else:
            normal = normal / np.linalg.norm(normal)
          vectors = pc - point
          all_distances = np.abs(np.dot(vectors, normal))
          inliers = all_distances <= max_dist
          n_inliers = np.sum(inliers)
          if n_inliers > n_best_inliers:
              n_best_inliers = n_inliers
              best_inliers = inliers
              if n_best_inliers > n_inliers_to_stop:
                  break
      print("plane: ", best_inliers)            # true/false array
      print("len plane: ", len(best_inliers))
      print("len pc   : ", len(pc))             # same len as above
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

# line_intersection((A, B), (C, D))
def line_intersection(line1, line2, dist = None):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def pt_on_line_seg(line_seg, dist):
    xdiff = (line_seg[0][0] - line_seg[1][0])
    ydiff = (line_seg[0][1] - line_seg[1][1])
    if xdiff == 0:
      return line_seg[0][0], line_seg[0][1]+dist
    slope = ydiff / xdiff
    dx = math.sqrt(dist * dist / (slope*slope+1))
    dy = slope * dx
    return line_seg[0][0]+dx, line_seg[0][1]+dy
    
def add_color_slow(pc1,pc2):
   # simple N*M + 2N algorithm: unacceptably slow
   min_dist = [[BIGNUM, 0] for i1, p1 in enumerate(pc1)]
   print("add_color")
   for i1, p1 in enumerate(pc1):
     for p2 in pc2:
       if distance_3d(p1,p2) < min_dist[i1][0]:
         min_dist[i1] = [distance_3d(p1,p2), p2[3]]
   pc3 = [[p1[0],p1[1],p1[2],min_dist[i1][1]] for i1, p1 in enumerate(pc1)]
   print("add_color2")
   return pc3

# pc1 should be an octomap
def add_color(pc1,pc2):
   print("add_color")
   # simple O(len(pc1) + len(pc2)) algorithm
   # TODO: replace following with computed numbers
   # min = [-0.16875, -0.14875, 0.46625]
   # max = [0.17125, 0.15125, 0.48875]
   # resolution = .0025
   # diff = [.34, .3, .0225]
   # xyz_dim = [136, 120, 9]

   def offset(pt):
     min = [-0.16875, -0.14875, 0.46625]
     xyz_dim = [136, 120, 9]
     resolution = .0025
     offst = []
     for i in range(3):
       offst.append(int((pt[i] - min[i]) / resolution + .5))
       if offst[i] < 0:
         offst[i] = 0
       if offst[i] >= xyz_dim[i]:
         offst[i] = xyz_dim[i] - 1
     return offst

   def center_pt(x,y,z):
     min = [-0.16875, -0.14875, 0.46625]
     resolution = .0025
     ctr = []
     for i in range(3):
       ctr.append(min[i] + (x * resolution))
     return ctr

   color = np.zeros((136, 120, 9)) 
   dist  = np.ones((136, 120, 9)) 
   for p in pc2:
     [x,y,z] = offset(p)
     d = distance_3d(p, center_pt(x,y,z)) 
     if d < dist[x,y,z]:
       dist[x,y,z] = d
       color[x,y,z] = p[3] 
   pc1_w_color = []
   for p_id, p in enumerate(pc1):
     x,y,z = offset(p)
     if dist[x,y,z] < 1:
       # print("x,y,z",x,y,z)
       p1 = pc1[p_id]
       pc1_w_color.append([p1[0],p1[1],p1[2],color[x,y,z]])
     else:
       found = False
       for i in range(5):
         p2_x = [x, max(x-i,0), min(x+i, 136-1)]
         p2_y = [y, max(y-i,0), min(y+i, 120-1)]
         p2_z = [z, max(z-i,0), min(z+i, 9-1)]
         for j1 in range(3):
           for j2 in range(3):
             for j3 in range(3):
               if dist[p2_x[j1], p2_y[j2], p2_z[j3]] < 1:
                 p1 = pc1[p_id]
                 colour = color[p2_x[j1], p2_y[j2], p2_z[j3]]
                 pc1_w_color.append([p1[0],p1[1],p1[2],colour])
                 found = True
                 break
             if found:
               break
           if found:
             break
         if found:
           break
       if not found:
         p1 = pc1[p_id]
         pc1_w_color.append([p1[0],p1[1],p1[2],0])
   return tuple(pc1_w_color)

