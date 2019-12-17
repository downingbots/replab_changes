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
        else:
          KP = None
        clusters.analyze_pc(pc, KP)

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

        grasp_conf = self.assign_grasp_confidence(grasps)
        clusters.assign_grasps(grasp_conf)
        return grasp_conf

    def assign_grasp_confidence(self, grasps):
        if grasps is None or len(grasps) == 0:
          return None
        # else base on interesting clusters
        # prioritize the top center of cluster
        else:
          prob = 1 / len(grasps)
          return [((g[0], g[1], g[2], g[3]), prob) for i,g in enumerate(grasps)]

    def evaluate_drop(self, grasp, grasps, position, e_i, a_i):
      # did the cluster move after being dropped?
      # ensure that the gripper is or can be closed

    def evaluate_grasp(self, grasp, grasps, position, e_i, a_i):
        # ISOLATE until YOLO is trained
        self.action_mode = "ISOLATE"     # ISOLATE, INTERACT, CURIOSITY

        eval_grasp_action = {}
        pose = self.get_pose()
        eval_grasp_action['EVA_POSE'] = [pose[0],pose[1],pose[2],pose[3]]
        eval_grasp_action['EVA_NEW_POSE'] = [pose[0],pose[1],pose[2],pose[3]]
        eval_grasp_action['EVA_SUCCESS'], eval_grasp_action['EVA_CLOSURE'] = self.widowx.eval_grasp(manual=manual)

        if eval_grasp_action['EVA_SUCCESS']:
          print("successful grasp: ",  eval_grasp_action['EVA_CLOSURE'])
          cluster_id = cluster.cluster_contains(grasp)
          if cluster.is_isolated():
            deg_horiz, flip = cluster.rotate_angle()
            if deg_horiz != None:
              eval_grasp_action['EVA_ACTION'] = "ROTATE"
              eval_grasp_action['EVA_DEG'] = deg_horiz
            elif flip:
              eval_grasp_action['EVA_ACTION'] = "FLIP"
            else:
              if cluster.get_action_mode() == "ISOLATED":
                cluster.analyze_object()
                cluster.set_action_mode("INTERACT")
                eval_grasp_action['EVA_ACTION'] = "RANDOM_DROP"
          elif self.action_mode == "ISOLATE":
            eval_grasp_action['EVA_ACTION'] = "ISOLATED_DROP"
            eval_grasp_action['EVA_REWARD'] = 1
          # else look cluster interaction using AIMED_DROP
          else
            eval_grasp_action['EVA_ACTION'] = "RANDOM_DROP"
            eval_grasp_action['EVA_REWARD'] = 1
        elif iterator == 2:
          eval_grasp_action['EVA_ACTION'] = "EVAL_WORLD_ACTION"
        elif iterator == 3:
          if eval_world_action['EWA_STATE'] = "UNCHANGED":
            eval_grasp_action['EVA_ACTION'] = "RETRY_GRASP"
            eval_grasp_action['EVA_REWARD'] = 0
            new_x += (random() * 2 - 1) * INCH
            new_y += (random() * 2 - 1) * INCH
            new_z += (random() * 2 - 1) * INCH
            # new_theta unchanged (?)
          elif eval_world_action['EWA_STATE'] = "MOVED":
            eval_grasp_action['EVA_ACTION'] = "NO_DROP"
            eval_grasp_action['EVA_REWARD'] = eval_world_action['REWARD'] 
        elif iterator == 4:
          r = random()
          if r < .25:
            # try a new grasp from list
            eval_grasp_action['EVA_ACTION'] = "GRASP"
          elif r < .5:
            eval_grasp_action['EVA_ACTION'] = "RETRY_GRASP"
            new_x += (random() * 2 - 1) * INCH
            new_y += (random() * 2 - 1) * INCH
            new_z += (random() * 2 - 1) * INCH
          elif r < .75:
            eval_grasp_action['EVA_ACTION'] = "EVAL_WORLD_ACTION"
          else:
            eval_grasp_action['EVA_ACTION'] = "PUSH"
            _,_,_, cur_theta = eval_grasp_action['EVA_POSE']
            eval_grasp_action['EVA_DEG'] = cur_theta
        elif iterator % 5 == 0 and iterator < 20:
          self.grasp_hist.append(grasp)
          eval_grasp_action['EVA_ACTION'] = "RETRY_GRASP"
          print("unsuccessful grasp: ", eval_grasp_action['EVA_CLOSURE'])
        elif iterator < ::
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

    def evaluate_world(self, grasp, grasps, position, e_i, a_i):
        success, closure = self.widowx.eval_grasp(manual=manual)
        # T, distances, i = icp(A, B)   # nxM matrices
        # look for differences
        return success

    def __init__(self, event_history):
        clusters = clusterState()
        recent_grasps = None
      

