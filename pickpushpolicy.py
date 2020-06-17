import numpy as np
from scipy.linalg import eigh

# from grasp_network import FullImageNet, PintoGuptaNet
from replab_core.config import *
from replab_core.utils import *
from utils_grasp import *
from config_grasp import *
from policy import *
from keypoint import *
from cluster import *
from world import *
# from replab_grasping.utils_grasp import *
# from replab_grasping.config_grasp import *
# from replab_grasping.policy import *
# from replab_grasping.keypoints import *
from scipy import spatial
from scipy.stats import linregress
import statistics
import math
import sys

import traceback

import torch.nn as nn
import torchvision.transforms as transforms

class PickPushCollection(Policy):

    def compare_states(self, state1, state2):
      pass

    def get_grasp_cluster(self, grasp):
      return self.world_clusters.in_any_obb([grasp[0],grasp[1],grasp[2]])

    '''
    Grasps the center of the object with theta perpendicular to the principal axis
    '''
    # def plan_grasp(self, rgb, pc):
    # 
    def plan_grasp(self, octomap, octomap_header):
        def take_z_axis(elem):
          # ARD
          return elem[2]
          # return -elem[2]

        ##################
        # CLUSTER ANALYSIS
        ##################
        sys.setrecursionlimit(20000)
        # pc_clusters is based on current octomap 
        pc_clusters = WorldState()
        if self.world_clusters != None:
          # enable optimizations based upon world state
          # print("w_clust", len(self.world_clusters.octobase))
          pc_clusters.copy_world_state_to_curr_pc(self.world_clusters)
        if pc_clusters.analyze_pc(octomap, octomap_header, self.min_samples) == False:
          self.min_samples -= 1
          return None
        # world_clusters is based on previous octomap 
        if self.world_clusters == None:
          self.world_clusters = WorldState()
          self.world_clusters.initialize_world(pc_clusters)
          print("w_clust", len(self.world_clusters.octobase))
        else:
          # update world_clusters to determine changes in octomap analysis
          self.world_clusters.integrate_current_pc_into_world(pc_clusters)
          print("w_clust", len(self.world_clusters.octobase))
          self.pc_cluster_history.append(pc_clusters)
          self.display_cluster_history(self.world_clusters.prev_target_w_cluster)
          if len(self.pc_cluster_history) > WORLD_HISTORY_LEN:
            # reached the end of the world
            del self.pc_cluster_history[0]

        DEBUG_OCTOMAP = False
        if DEBUG_OCTOMAP:
          # skip grasping
          return None
        ###################
        # GRASPING ANALYSIS
        ###################
        # octoclusters = self.world_clusters.octoclusters
        grasps = None     #  x, y, z, theta, probabilities = grasp
        octoclusters = []
        cluster_grasp_cnt = []
        grasp_cluster = []
        for i in range(len(self.world_clusters.clusters)):
          # max grasps per cluster
          if i not in self.world_clusters.possible_octobase:
            if self.world_clusters.active(i):
              # no colors in neighbor computation
              octoclusters.append(np.array(self.world_clusters.clusters[i].cluster['shape'])[:,:3])
          cluster_grasp_cnt.append(0)
        for c_id, c in enumerate(octoclusters):
          grasp_pc = sorted(c, key=take_z_axis)
          print("c_id, len grasp_pc: ", c_id, len(grasp_pc))
          kdtree = spatial.KDTree(grasp_pc)
          evaluated  = None
          success = False
          skipped_evaluated = False
          for p_i, p in enumerate(grasp_pc):
            if evaluated is not None and p_i in evaluated:
              # if not skipped_evaluated:
              #   print("prev eval: ", len(evaluated))
              skipped_evaluated = True
              continue
            else:
              skipped_evaluated = False
            # print("p[2] =", p[2])
            
            # returns a list of the indices of the neighbors of p
            # r ~ 1.5inches  (maybe make smaller?)
            neighbors = kdtree.query_ball_point(p, r=GRIP_EVAL_RADIUS)
  
            # only consider points that can potentially fit in gripper.
            # grip height ~ 1inch
            pc2 = [grasp_pc[n] for i, n in enumerate(neighbors) 
                   if abs(p[2] - grasp_pc[n][2]) <= GRIPPER_HEIGHT]
            if len(pc2) == 0:
              continue
            x = [p2[0] for i,p2 in enumerate(pc2)]
            y = [p2[1] for i,p2 in enumerate(pc2)]
            z = [p2[2] for i,p2 in enumerate(pc2)]
  
            # points required to fit in gripper.
            # MIN_GRIP_HEIGHT vs GRIPPER_HEIGHT
            pc3 = [grasp_pc[n] for i, n in enumerate(neighbors) 
                   if abs(p[2] - grasp_pc[n][2]) <= MIN_GRIP_HEIGHT]
            if len(pc3) < MIN_NEIGHBOR_THRESH:
              print("Min Neighbors: ", len(pc3), len(pc2))
              continue
            z_base = self.world_clusters.get_base_z(p[0],p[1])
            if len(pc2) - len(pc3) < MIN_NEIGHBOR_THRESH and (z_base - p[2] > MIN_GRIP_HEIGHT):
              if len(pc2) > len(pc3):
                print("MIN_GRIPPER_HEIGHT neighbor dif:", (len(pc2) - len(pc3)),(z_base - p[2]))
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
            oob_thresh = OOB_THRESH
            # find gripper orientation
            if len(x3) <= 1:
              degrees = 0
              # print("degrees", degrees)
              success = False
              num = num+1
            else:
              success = True
              try:
                # linear regression needed to find gripper orientation
                # if there's a wide gripping surface. However, a cicularg
                # surface is acceptable. Therefore no need for R2 or P checks.
                slope, intercept, r_value, p_value, std_err = linregress(x3, y3)
              except:
                print("Unexpected error:", sys.exc_info()[0])
                success = False
              if math.isnan(slope):
                success = False
                print("Slope is NaN")

#              # r-squared: betw bad (0) to good fit (1)
#              # According to Cohen (1992) r-square value:
#              #   .12 or below indicate low, 
#              #   between .13 to .25 values indicate medium,
#              #   .26 or above and above values indicate high effect size. 
#              R2_TARGET = .25
#              r2 = r_value * r_value
#              if r2 < R2_TARGET:
#                success = False
#                print("R2 failed:", r2)
#
#              # two-sided p-value null hypothesis is that the slope is zero.
#              # check correlation is significant (betw 0 & 1)
#              # is there a significant relationship between the independent 
#              # and dependent variables?
#              # sample pvalue_target=0.05, .1
#              pvalue_target=0.1
#              if p_value > pvalue_target:
#                success = False
#                print("Pval failed:", p_value)

              if success:
                angle = math.atan(slope)             # slope angle in radians
                degrees = math.degrees(angle)        # slope angle in degrees
                # print("degrees", degrees, " = angle", angle, " slope ", slope, "x3",x3,y3)
                # thetas.append(np.arctan2(eigv[1], eigv[0]) % np.pi)
                # to convert from degrees to radians, multiply by pi/180.
                theta = angle
    
                # make sure it's a feasible grasp
                # success = True
                expand_grip_height = True
                for i, x1 in enumerate(x):
                  d = shortest_distance_from_line( x1, y[i], slope, -1, intercept)
                  gw = GRIPPER_WIDTH/2    # 0.01143
                  if (d > gw):
                    # too wide to grip
                    # z=.25 or .00625 is deep enough, anything deeper is gravy
                    if abs(p[2] - z[i]) >= MIN_GRIP_HEIGHT: # required to grip
                      num = num+1
                      # print("OOB:", i, "x:", round(x1,5), "y:", round(y[i],5), "d",round(d,5), "slope", round(slope,5), "inter", round(intercept,5), "zdif", round(abs(p[2] - z[i]),5) )
                      if num > oob_thresh:
                        success = False
                    else:
                      # can't grip farther than this, but doesn't eliminate grip
                      # ARD: was min????
                      max_grip_height = max(abs(p[2] - z[i]), max_grip_height)
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
                g_c_id = self.world_clusters.in_any_obb([g[0],g[1],g[2]])
                if g_c_id != None:
                  cluster_grasp_cnt[g_c_id] += 1
                  # self.world_clusters.clusters[g_c_id].cluster['grasps'].append(g)
                  if cluster_grasp_cnt[g_c_id] < 5:
                    # print("grasp in cluster",g_c_id)
                    grasps = []
                    grasps.append(g)
                    grasp_cluster.append(g_c_id)
            else:
              if success:
                g = [round(x_mean, precision), round(y_mean, precision), round(z_grip, precision), round(theta, precision)]
                g_c_id = self.world_clusters.in_any_obb([g[0],g[1],g[2]])
                if g_c_id != None:
                  if g not in grasps:
                    cluster_grasp_cnt[g_c_id] += 1
                    if cluster_grasp_cnt[g_c_id] < 5:
                      grasps.append(g)
                      grasp_cluster.append(g_c_id)
                      # print("Grasp x: ", x_mean, " y: ", y_mean, " deg: ", theta, " z: ", z_grip)
            if evaluated is None:
              evaluated = []
            for n_i, n in enumerate(neighbors):
              dist = distance_3d(p, grasp_pc[n])
              if dist < GRIP_EVAL_RADIUS/2 and n not in evaluated:
                evaluated.append(n)
        ## END PER-CLUSTER GRASP ANALYSIS LOOP ##

        grasp_conf = self.assign_grasp_confidence(grasps, grasp_cluster)
        # if len(self.world_clusters.clusters) > 0:
        self.world_clusters.assign_grasps(grasp_conf)
        # if len(pc_clusters.clusters) > 0:
          # pc_clusters.assign_grasps(grasp_conf, grasp_clust)
        return grasp_conf

    def assign_grasp_confidence(self, grasps, grasp_clust):
        def take_z_axis(elem):
          # ARD
          return elem[1]
          # return -elem[1]

        if grasps is None or len(grasps) == 0:
          return None
        # else base on interesting clusters
        # prioritize the top center of cluster
        # ARD TODO: eventually replace by: 
        # self.world_clusters.world_grasp_probabilities(..)
        # else:
        #   prob = 1 / len(grasps)
        #   return [[[g[0], g[1], g[2], g[3]], prob] for i,g in enumerate(grasps)]
        #
        tot = 0
        idx = [[0,0] for i in range(len(grasps))]
        for j,g in enumerate(grasps):
          idx[j][0] = j
          idx[j][1] = g[2]
          tot += j

        # sort by highest grasp
        idx = sorted(idx, key=take_z_axis)
        # combine cluster of grasps into single grasp with middle

        # idx2 = self.world_clusters.reorder_by_cluster_xy_center(grasps, idx)
        # idx2 = self.world_clusters.reorder_by_cluster_xy_center(grasps, idx)

        prob = [0.0 for i in range(len(grasps))]
        sum = 0
        for j in range(len(grasps)):
          # sum = j
          i = idx[j][0]
          if tot > 0:
            prob[i] = 1.0 * (len(grasps) - j) / tot
          else:
            prob[i] = 0
          print(i," grasp height ", idx[j][1], " prob ",prob[i], " clust", grasp_clust[idx[j][0]])

        # assign probability weighted towards x-y center of a cluster
       
        # prune those too close to higher-probability grasps
        
        retgrasp = [[[g[0], g[1], g[2], g[3]], prob[i]] for i,g in enumerate(grasps)]
        # print("retgrasp:", len(retgrasp), retgrasp)
        return retgrasp


    def evaluate_drop(self, grasp, grasps, position, e_i, a_i):
      # did the cluster move after being dropped?
      # ensure that the gripper is or can be closed
      pass

    def callibration_method(method):
      callibration = method

    def display_cluster_history(self, w_cluster_id):
        import rospy

        print("clear cluster and obb for publish_history")
        # rospy.sleep(5)
        for pc_hist_id, pc in enumerate(self.pc_cluster_history):
          print("print history of w_cluster:", pc_hist_id, w_cluster_id)
          pc.publish_pc_cluster(w_cluster_id)
          # rospy.sleep(5)

    def evaluate_grasp_target(self, octomap, grasp):
        print("evaluate grasp target")
        cluster_id = self.world_clusters.cluster_contains(grasp)
        if cluster_id == None:
          ret = False
        else:
          ret = self.world_clusters.did_cluster_move(octomap, cluster_id, self.world_clusters)
          print("did cluster move?", cluster_id, ret)
        eval_grasp_action = {}
        eval_grasp_action['EVA_SUCCESS'] = ret
        # self.display_cluster_history(cluster_id)
        return eval_grasp_action

    def clear_target_grasp(self):
        self.world_clusters.clear_target_grasp()

    def set_target_grasp(self, grasp, action):
        g_c_id = self.world_clusters.in_any_obb(grasp)
        self.world_clusters.set_target_grasp(g_c_id, grasp, action)

    def evaluate_grasp(self, w_c_id, action, grasp, grasps, pose, joints, e_i=None, a_i=None):
        # ISOLATE until YOLO is trained
        self.action_mode = "ISOLATE"     # ISOLATE, INTERACT, CURIOSITY

        eval_grasp_action = {}
        eval_grasp_action['EVA_ACTION'] = "RANDOM_DROP"
        eval_grasp_action['EVA_POSE'] = [pose[0],pose[1],pose[2],pose[3]]
        eval_grasp_action['EVA_NEW_POSE'] = [pose[0],pose[1],pose[2],pose[3]]
        # ARD: widowx should not be called by policy directly. OK in collect.py
        # eval_grasp_action['EVA_SUCCESS'], eval_grasp_action['EVA_CLOSURE'] = self.widowx.eval_grasp(manual=manual)
        # eval_grasp_action['EVA_SUCCESS'], eval_grasp_action['EVA_CLOSURE'] = self.widowx.eval_grasp(manual=manual)
        # print("eval_grasp: ", threshold, manual)
        # GRIPPER_CLOSED = [.003, .003]
        gripper_gap = joints[0] - np.array(GRIPPER_CLOSED[0])
        threshold=.0003
        if (gripper_gap > threshold):
          print("eval_grasp:", gripper_gap, joints[0], pose[0])
          eval_grasp_action['EVA_SUCCESS'] = True
          eval_grasp_action['EVA_CLOSURE'] = gripper_gap
        else:
          eval_grasp_action['EVA_SUCCESS'] = False
          eval_grasp_action['EVA_CLOSURE'] = 0
        # print("eval_grasp: ", threshold, manual)
        self.world_clusters.clusters[w_c_id].add_to_history(action, result)
        return eval_grasp_action

    # currently being done in "goal_plan"
    def post_grasp_action(self, grasp, grasps, pose, joints, e_i=None, a_i=None):
        iterator = 1          # ARD: huh? Needs to be fleshed out
        if eval_grasp_action['EVA_SUCCESS'] and False:
          print("successful grasp: ",  eval_grasp_action['EVA_CLOSURE'])
          cluster_id = self.world_clusters.cluster_contains(grasp)
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
          else:
            eval_grasp_action['EVA_ACTION'] = "RANDOM_DROP"
            eval_grasp_action['EVA_REWARD'] = 1
        elif iterator == 2:
          eval_grasp_action['EVA_ACTION'] = "EVAL_WORLD_ACTION"
        elif iterator == 3:
          if eval_world_action['EWA_STATE'] == "UNCHANGED":
            eval_grasp_action['EVA_ACTION'] = "RETRY_GRASP"
            eval_grasp_action['EVA_REWARD'] = 0
            new_x += (random() * 2 - 1) * INCH
            new_y += (random() * 2 - 1) * INCH
            new_z += (random() * 2 - 1) * INCH
            # new_theta unchanged (?)
          elif eval_world_action['EWA_STATE'] == "MOVED":
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
        # elif iterator < ::
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

    def evaluate_world(self, grasp=None, grasps=None, position=None, e_i=None, a_i=None):
        # ARD: widowx should not be called by policy directly. OK in collect.py
        # success, closure = self.widowx.eval_grasp(manual=manual)
        # T, distances, i = icp(A, B)   # nxM matrices
        # look for differences
        # return success
        return True

    # get_octomap and subsequent analysis takes a lot of time.
    # each "move" does one grasp attempt per cluster per get_octomap
    def init_move(self, grasps, confidences, initial_grasp):
        self.visited_c_id = []
        self.unvisited_conf   = [c for i, c in enumerate(confidences)]
        self.unvisited_grasps = [i for i in range(len(confidences))]
        del self.unvisited_grasps[initial_grasp]
        sum_conf = 0
        for c in self.unvisited_conf:
          sum_conf += c
          if sum_conf > 0:
            self.unvisited_conf /= sum_conf

    def next_grasp_in_move(self, grasps, confidences):
        # find next grasp for a cluster that has not been attempted
        # during this move (get_octomap)
        grasp = None
        if len(self.unvisited_grasps) <= 0:
          return None
        while len(self.unvisited_grasps) > 0:
          selected = np.random.choice(self.unvisited_grasps, p = (self.unvisited_conf))
          print("selected", selected, len(self.unvisited_conf), self.unvisited_grasps)
          if selected not in visited_grasps:
            self.unvisited_grasps.remove(selected)
            self.unvisited_conf = [c for i, c in enumerate(confidences) if i in self.unvisited_grasps]
            sum_conf = 0 
            for c in self.unvisited_conf:
              sum_conf += c
            if sum_conf > 0:
              self.unvisited_conf /= sum_conf
          else:
            continue
          grasp = grasps[selected][0]
          c_id = policy.get_grasp_cluster(grasp)
          if c_id == None or c_id in self.visited_c_id:
            grasp = None
            continue   # only visit a c_id once per image
          else:
            self.visited_c_id.append(c_id)
            break
        return grasp

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def __init__(self, event_history=None):
        self.min_samples = CLUSTER_MIN_SZ
        self.world_clusters = None
        self.pc_cluster_history = []
        self.recent_grasps = None
        self.callibration = "none"
        self.visited_c_id = None
        self.unvisited_conf   = None
        self.unvisited_grasps = None
        self.goal_state = None
      


