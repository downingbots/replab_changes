import numpy as np
from scipy.linalg import eigh

# from grasp_network import FullImageNet, PintoGuptaNet
from replab_core.config import *
from replab_core.utils import *
from utils_grasp import *
from config_grasp import *
from policy import *
from keypoint import *
# from replab_grasping.utils_grasp import *
# from replab_grasping.config_grasp import *
# from replab_grasping.policy import *
# from replab_grasping.keypoints import *
from scipy import spatial
from scipy.stats import linregress
import statistics
import math

import traceback

import torch.nn as nn
import torchvision.transforms as transforms

class PickPushCollection(Policy):

    def analyze_pc_clusters(self, pc, KP):
        db1 = DBSCAN(eps=.001, min_samples=3,
                     n_jobs=-1).fit(pc)
        # Number of clusters in labels, ignoring noise if present.
        n1_clusters = len(set(db1.labels_)) - (1 if -1 in db1.labels_ else 0)
        print("DBSCAN: # ", n1_clusters)
        cluster_centers = []
        cluster_shapes = []
        cluster_KPs = []
        kp = KP.get_kp() 

        for cluster in set(db1.labels):
          if cluster != -1:
            running_sum = np.array([0.0, 0.0, 0.0])
            counter = 0
            cluster_kp = []
            cluster_shape = []
            for i in range(pc.shape[0]):
                if db1.labels[i] == cluster:
                    running_sum += pc[i]
                    counter += 1
                    cluster_shape.append(pc[i])
                    if [pc[i][0], pc[i][1]] in kp:
                      cluster_kp.append(pc[i])
                      print("kp found in cluster")
            center = running_sum / counter
            cluster_centers.append(center)
            cluster_KPs.append(found_kp)
            cluster_shapes.append(cluster_shape)
        return [pc, KP, counter, cluster_centers, cluster_KPs, cluster_shapes]

    def compare_clusters(self, state1, state2):
      pass

    def compare_states(self, state1, state2):
      pass

    '''
    Grasps the center of the object with theta perpendicular to the principal axis
    '''
    # def plan_grasp(self, rgb, pc):
    def plan_grasp(self, rgb, pc):
        def take_z_axis(elem):
          return elem[2]
        pc1 = sorted(pc, key=take_z_axis)
        # print("old pc[0]: ", pc[0])
        # print("new pc[0]: ", pc1[0])

        kdtree = spatial.KDTree(pc1)
        if COMPUTE_KEYPOINT:
          # return np.concatenate([self.rgb, depth], axis=2)
          # rgb = np.split([self.rgb, depth], axis=2)
          # rgb = rgbd[:, :, :3].astype(np.uint8)
          # print("len rgbd ", len(rgbd))  # 480
          # print("len rgbd[0] ", len(rgbd[0]))  # 640
          # print("rgbd[0] ", rgbd[0][0])        
          # print("rgbd[0][0] ", rgbd[0][0][0])        
          # rgb = []
          # for i in range(480):
          #   for j in range(640):
          #     rgb.append((rgbd[i][j][0], rgbd[i][j][1], rgbd[i][j][3]))
          KP = Keypoints(rgb)
          KP.publish_img(rgb)
          # self.keypoints = KP.get_kp
          kp_pc_points = KP.kp_to_3d_point(pc)
          KP.publish_pc(pc)

        evaluated  = None
        grasps = None     #  x, y, z, theta, probabilities = grasp
        success = False
        skipped_evaluated = False
        # base_z = compute_z_sectors(pc1)
        for p_i, p in enumerate(pc1):

          if evaluated is not None and p_i in evaluated:
            # if not skipped_evaluated:
            #   print("prev eval: ", len(evaluated))
            skipped_evaluated = True
            continue
          else:
            skipped_evaluated = False
          ##
          ## done earlier: doing again will be 2*MIN_OBJ_HEIGHT
          ##
          # sect = get_sector(p[0],p[1])
          # if (p[2] > base_z[sect] - MIN_OBJ_HEIGHT):
          #   print("TOO CLOSE TO GROUND")
          #   continue
          
          # returns a list of the indices of the neighbors of p
          neighbors = kdtree.query_ball_point(p, r=GRIP_EVAL_RADIUS)

          # see if kp neighbor
          if FAVOR_KEYPOINT:
            kp_neighbor = False
            if kp_pc_points is not None:
              for n_i,n in enumerate(neighbors):
                for kp_i, kp in enumerate(kp_pc_points):
                  if pc1[n][0] == kp[0] and pc1[n][1] == kp[1] and pc1[n][2] == kp[2]:
                    print("KP NEIGHBOR: ", p)
                    kp_neighbor = True
                    break
                if kp_neighbor:
                  break

          pc2 = [pc1[n] for i, n in enumerate(neighbors) 
                 if abs(p[2] - pc1[n][2]) <= GRIPPER_HEIGHT]
          if len(pc2) == 0:
            continue
          x = [p2[0] for i,p2 in enumerate(pc2)]
          y = [p2[1] for i,p2 in enumerate(pc2)]
          z = [p2[2] for i,p2 in enumerate(pc2)]

          pc3 = [pc1[n] for i, n in enumerate(neighbors) 
                 if abs(p[2] - pc1[n][2]) <= MIN_GRIP_HEIGHT]
          if len(pc3) < MIN_NEIGHBOR_THRESH:
            print("Min Neighbors: ", len(pc3), len(pc2))
            continue
          # x_mean = sum(pc3[0])/len(pc3[0])
          # y_mean = sum(pc3[1])/len(pc3[1])
          x3 = [p3[0] for i3,p3 in enumerate(pc3)]
          y3 = [p3[1] for i3,p3 in enumerate(pc3)]
          x_mean = sum(x3)/len(x3)
          y_mean = sum(y3)/len(y3)
          # if evaluated is not None and pc3 is not None and evaluated is not None:
          #   print("#neighbors: ", len(neighbors), " len(pc3): ", len(pc3), " prev eval: ", len(evaluated))
          z_grip = p[2] + MIN_GRIP_HEIGHT

          num = 0
          max_grip_height = MIN_GRIP_HEIGHT   # initialize before computing
          if FAVOR_KEYPOINT and kp_neighbor:
            oob_thresh = KP_OOB_THRESH
          else:
            oob_thresh = OOB_THRESH
          # find gripper orientation
          if len(x3) <= 1:
            degrees = 0
            success = False
            num = num+1
          else:
            slope, intercept, r_value, p_value, std_err = linregress(x3, y3)
            angle = math.atan(slope)             # slope angle in radians
            degrees = math.degrees(angle)        # slope angle in degrees
            # thetas.append(np.arctan2(eigv[1], eigv[0]) % np.pi)
            # to convert from degrees to radians, multiply by pi/180.
            theta = angle

            # make sure it's a feasible grasp
            success = True
            expand_grip_height = True
            for i, x1 in enumerate(x):
              d = shortest_distance_from_line( x1, y[i], slope, -1, intercept)
              if FAVOR_KEYPOINT and kp_neighbor:
                gw = GRIPPER_WIDTH
              else:
                gw = GRIPPER_WIDTH/2    # 0.01143
              if (d > gw):
                # too wide to grip
                # z=.25 or .00625 is deep enough, anything deeper is gravy
                if abs(p[2] - z[i]) <= MIN_GRIP_HEIGHT: # required to grip
                  num = num+1
                  # print("OOB:", i, "x:", round(x1,5), "y:", round(y[i],5), "d",round(d,5), "slope", round(slope,5), "inter", round(intercept,5), "zdif", round(abs(p[2] - z[i]),5) )
                  if num > oob_thresh:
                    success = False
                    if FAVOR_KEYPOINT and kp_neighbor:
                      print("failed: gripper distance: ", d)
                      print("failed: depth distance: ", abs(p[2] - z[i]))
                  # break
                else:
                  # can't grip farther than this, but doesn't eliminate grip
                  max_grip_height = min(abs(p[2] - z[i]), max_grip_height)
                  # can't grip farther than this
                  expand_grip_height = False
                  z_grip = min(z_grip, z[i])
                  # z_grip = max(z_grip, p[2] + MIN_GRIP_HEIGHT)
                  if (max_grip_height >  MIN_GRIP_HEIGHT and
                     max_grip_height <= GRIPPER_HEIGHT):
                    z_grip = min(z_grip, p[2] + max_grip_height)
                  elif max_grip_height > GRIPPER_HEIGHT:
                    z_grip = p[2] + GRIPPER_HEIGHT
              elif expand_grip_height:
                  max_grip_height = max(abs(p[2] - z[i]), max_grip_height)
                  if max_grip_height <= GRIPPER_HEIGHT:
                    z_grip = max(z_grip, z[i])
                  else:
                    z_grip = p[2] + GRIPPER_HEIGHT
              z_grip = max(z_grip, p[2] + MIN_GRIP_HEIGHT)
              z_grip = min(z_grip, p[2] + GRIPPER_HEIGHT)

          # z_grip = z_grip + Z_PLATFORM
          if num <= oob_thresh:
            pass
            # print("SUCCESS  SUCCESS  SUCCESS  SUCCESS  SUCCESS  SUCCESS")
            # print("num OOB: ", num)
          if grasps is None:
            if success:
              precision = 5
              g = [round(x_mean, precision), round(y_mean, precision), round(z_grip, precision), round(theta, precision)]
              grasps = []
              grasps.append(g)
          else:
            if success:
              g = [round(x_mean, precision), round(y_mean, precision), round(z_grip, precision), round(theta, precision)]
              if g not in grasps:
                grasps.append(g)
              print("Grasp x: ", x_mean, " y: ", y_mean, " deg: ", theta, " z: ", z_grip)
          if evaluated is None:
            evaluated = []
          for n_i, n in enumerate(neighbors):
            if n not in evaluated:
              evaluated.append(n)
        return self.assign_grasp_confidence(grasps)

    def assign_grasp_confidence(self, grasps):
        if grasps is None or len(grasps) == 0:
          return None
        else:
          prob = 1 / len(grasps)
          return [((g[0], g[1], g[2], g[3]), prob) for i,g in enumerate(grasps)]

    def __init__(self):
        recent_grasps = None
      


class PickPushCurious(Policy):
    '''
    Implementation of Pinto 2016
    Details found here: https://arxiv.org/pdf/1509.06825.pdf
    '''

    def __init__(self, model_path=None, heightmaps=False):
        self.net = PintoGuptaNet(depth=False, binned_output=True).cuda()
        self.net = nn.DataParallel(self.net).cuda()
        self.net.load_state_dict(torch.load(model_path),strict=False)
        self.net.eval()
        self.resize = make_resize_rgb(227, 227)

        self.K = DEPTH_K

        self.cm = CALIBRATION_MATRIX

        self.inv_cm = np.linalg.inv(self.cm)

    def calculate_crops(self, grasps):
        grasps = np.concatenate([grasps, np.ones((len(grasps), 1))], axis=1)
        camera_points = np.dot(self.inv_cm, grasps.T)[:3]
        camera_points = (camera_points.T + np.array([0.026, 0.001, 0.004])).T
        pixel_points = np.dot(self.K, camera_points / camera_points[2:])[:2].T
        return pixel_points.astype(int)

    def plan_grasp(self, rgbd, pc, num_grasps=256, batch_size=128):
        rgb = rgbd[:, :, :3].astype(np.uint8)
        _, labels = compute_blobs(pc)

        blobs = []
        for label in set(labels):
            if label == -1:
                continue
            blob_points = pc[labels == label]
            index = np.random.randint(0, len(blob_points))
            blobs.append(blob_points[index])

        all_grasps = []
        all_probabilities = []
        all_crops = []

        for blob in blobs:
            blob = np.concatenate([blob, [0.]], axis=0)
            blob[2] = Z_MIN

            candidates = []
            probabilities = []
            cropss = []

            for i in range(num_grasps // batch_size):
                noise = np.random.uniform([-XY_NOISE, -XY_NOISE, -.02, -1.57], [XY_NOISE, XY_NOISE, 0.0, 1.57],
                                          (batch_size, 4))
                grasps = noise + blob
                candidates.append(grasps)

                crops = self.calculate_crops(grasps[:, :3])

                cropped = []
                for crop in crops:
                    img = crop_image(rgb, crop, 48)
                    cropped.append(self.resize(img))

                cropss.append(crops)

                rgbs = torch.stack(cropped).cuda()
                grasps = torch.tensor(grasps, dtype=torch.float).cuda()

                output = self.net.forward((rgbs, None, grasps))

                probabilities.extend([sigmoid(k)
                                      for k in output.detach().cpu().numpy()])
            candidates = np.concatenate(candidates, axis=0)
            best_indices = np.argsort(probabilities)[-5:]
            best_index = np.random.choice(best_indices)
            cropss = np.concatenate(cropss, axis=0)

            all_crops.append(cropss[best_index])
            all_grasps.append(candidates[best_index])
            all_probabilities.append(probabilities[best_index])

        all_grasps = np.array(all_grasps)

        return [(grasp, all_probabilities[i]) for i, grasp in enumerate(all_grasps)]
