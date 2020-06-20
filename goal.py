from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from cluster import *
from world import *
from utils_grasp import *

class GoalState():
  def __init__(self, goal_octomap, goal_header):
    def take_z_axis(elem):
      return elem[2]

    x,y,z,th = 0,1,2,3
    # compute once as goal octomap doesn't change over time 
    self.goal_octomap_tuples = goal_octomap
    self.goal_header = goal_header
    self.goal_octomap_pc = [list(pt) for pt in goal_octomap]
    self.sorted_goal_octomap = sorted(self.goal_octomap_pc, key=take_z_axis)
    print("len goal_octomap_pc:", len(self.goal_octomap_pc))
    print("len goal_octomap_pc[0]:", len(self.goal_octomap_pc[0]))
    base_dist = self.sorted_goal_octomap[0][z]
    # self.goal_octomap_no_rgb = []
    goal_non_base_idx = None
    for pt_i, pt in enumerate(self.sorted_goal_octomap):
      # remove those withing .2 inches of bottom
      if pt[z] <= base_dist - (3.1 * OCTOMAP_RESOLUTION):
        self.goal_octobase.append(pt)
      else:
        # if goal_non_base_idx == None:
        goal_non_base_idx = pt_i
        # self.goal_octomap_no_rgb.append([pt[0],pt[1],pt[2]])
        break

    self.goal_octomap_no_rgb = np.array(self.sorted_goal_octomap)[goal_non_base_idx:,:3]
    self.goal_octomap = self.sorted_goal_octomap[goal_non_base_idx:]
    self.kdtree = spatial.KDTree(self.goal_octomap_no_rgb)

    self.goal_move   = [] # set of [[state, action, result],[state,action,rslt] ...]
    self.goal_state  = [] 
    self.goal_action = [] # [c_id, [action], succ/fail, dist_moved]
    self.goal_result = [] 
    self.min_samples = CLUSTER_MIN_SZ
    self.grasp_history  = []
    self.action_history = []


    # for computing scores of world state
    self.close_map = []
    self.far_map   = []
    self.unmapped  = []
    self.octobase  = []  # simple computation based on full world octomap
    self.cluster_move_history = {}
    self.MAX_COLOR_DIF = 5  # delta_e loses meaning above this value
    self.DIST_WEIGHT   = 10
    self.goal_header = goal_header
    # analyze the goal octomap to find clusters
    self.goal_clusters = WorldState()
    print("###########")
    print("Analyzing Goals")
    if self.goal_clusters.analyze_pc(self.goal_octomap_pc, goal_header, self.min_samples) == False:
      print("Analyzing Goals Failed!")
      exit()
    for g_c_id,g_c in enumerate(self.goal_clusters.clusters):
      if g_c_id in self.goal_clusters.possible_octobase:
        self.goal_clusters.w_poss_base_pts.append(g_c.cluster['shape'])
        # self.goal_clusters.clusters[g_c_id].cluster['shape'] = []
        self.goal_clusters.clusters[g_c_id].cluster['status'] = "BASE"
      elif g_c.cluster['status'] == "INACTIVE" and len(g_c.cluster['shape']) > 0:
        self.goal_clusters.clusters[g_c_id].cluster['status'] = "ACTIVE"
    print("Goal Analysis Complete")
    print("###########")

  def color_score(self, goal_color, w_color):
    rgb_r = int(math.floor(0.5 + w_color * 255))
    rgb_g = int(math.floor(0.5 + w_color * 255))
    rgb_b = int(math.floor(0.5 + w_color * 255))
    goal_rgb_r = int(math.floor(0.5 + goal_color * 255))
    goal_rgb_g = int(math.floor(0.5 + goal_color * 255))
    goal_rgb_b = int(math.floor(0.5 + goal_color * 255))
    curr_rgb = sRGBColor(rgb_r,rgb_g,rgb_b,is_upscaled=True)
    goal_rgb = sRGBColor(goal_rgb_r,goal_rgb_g,goal_rgb_b,is_upscaled=True)
    curr_lab = convert_color(curr_rgb, LabColor)
    goal_lab = convert_color(goal_rgb, LabColor)
    delta_e = delta_e_cie2000(curr_lab, goal_lab)
    # reward per pixel; lower is better
    delta_e_score = self.MAX_COLOR_DIF - max(delta_e, self.MAX_COLOR_DIF)
    return delta_e_score

  # works without cluster analysis
  # lowest score is best score
  def compute_goal_score(self, octomap_pc):
    def take_z_axis(elem):
      return -elem[2]

    [x,y,z,rgb_idx] = [0,1,2,3]
    sorted_octomap = sorted(octomap_pc, key=take_z_axis)
    self.close_map = []
    self.far_map   = []
    self.unmapped  = []
    self.octobase  = []
#    seen           = {}
#    sum_depth = 0
#    num_depth = 0
#    for pt_i, pt in enumerate(sorted_octomap):
#      if pt_i >= 10:
#        sum_depth += pt[z]
#        num_depth += 1
#      if pt_i >= 110:
#        break
#    base_dist = 0
#    if num_depth > 0:
#      base_dist = sum_depth / num_depth
#      print("base_dist: ",base_dist)
    non_base_idx = 0
    for pt_i, pt in enumerate(sorted_octomap):
      # remove those within .2 inches of bottom
      # don't want to require full image analysis to estimate the base
      # if pt[z] >= base_dist - (2.1 * OCTOMAP_RESOLUTION):
      # if self.goal_clusters.base_z[get_sector(pt[x], pt[y])] < pt[z]:
      if self.goal_clusters.base_z[get_sector(pt[x], pt[y])] - (2.1 * OCTOMAP_RESOLUTION) < pt[z]:
        self.octobase.append(pt)
      else:
        print("base_z:", self.goal_clusters.base_z[get_sector(pt[x], pt[y])])
        print("pt[z]: ", pt[z])
        non_base_idx = pt_i
        break

    sum_dist  = 0
    sum_color = 0
    sw_pcw_pcum_unmapped = 0
    max_dist  = 0
    print("non_base_idx:",non_base_idx)
    print("len sorted_octomap:",len(sorted_octomap))
    np_sort_map = (np.array(sorted_octomap[non_base_idx:]))
    print("len filtered sorted_octomap:",len(np_sort_map))
    np.random.shuffle(np_sort_map)
    print("random w_pc len:", len(np_sort_map))
    goal_seen = {}
    for pt_i, pt in enumerate(np_sort_map):
      pnt = [pt[x], pt[y], pt[z]]
      pnt_rgb = pt[rgb_idx]
      # nearest neighbors
      dist, ind = self.kdtree.query(pnt, k=6)                # doctest: +SKIP
      # print(ind)  # indices of 3 closest neighbors [0 3 1]
      # print(dist)  # distances to 3 closest neighbors [ 0. 0.196 0.294]

      found = False
      for i in range(len(ind)): 
          goal_pt = self.goal_octomap_no_rgb[ind[i]]
          # if not goal_seen[tuple(goal_pt)]:
          if not tuple(goal_pt) not in goal_seen:
            if dist[i] < 2 * OCTOMAP_RESOLUTION:
              self.close_map.append([pt_i, ind[i], dist[i]])
              goal_rgb = self.goal_octomap[ind[i]][rgb_idx]
              sum_color += color_score(goal_rgb, pnt_rgb)
            else:
              self.far_map.append([pt_i, ind[i], dist[i]])
              sum_dist = self.MAX_COLOR_DIF + dist[i]
              max_dist = max(max_dist, dist[i]) * self.DIST_WEIGHT
            found = True
            goal_seen[tuple(pnt)] = True
            break
      if not found:
        self.unmapped.append(pt_i)
    sum_unmapped = len(self.unmapped) * (max_dist + self.MAX_COLOR_DIF)
    total_score = sum_color + sum_dist + sum_unmapped
    return total_score

  def get_goal_info(self, policy, w_c_id, g_c_id):
    w_obb = policy.world_clusters.clusters[w_c_id].cluster['obb']
    g_obb = self.goal_clusters.clusters[g_c_id].cluster['obb']
    w_loc = w_obb.centroid
    g_loc = g_obb.centroid
    dist = distance_3d(w_loc, g_loc)
    return [dist, w_loc, g_loc, w_obb, g_obb]

  def goal_state.record_plan_result(pick_result, action_completed):
    # [w_cid, "GRASP", grasp, False]
    self.grasp_history.append(pick_result)
    self.action_history.append(action_completed)

  def previously_tried_grasp(self, gs_w_cid, grasp):
    # [w_cid, "GRASP", grasp, False]
    for grasp_result in reversed(self.grasp_history):
        [gr_w_cid, gr_action, gr_grasp, gr_result] = prev_grasp
        if gr_action != "GRASP" or gr_w_cid != gs_w_cid:
          continue
        if distance_3d(grasp, gr_grasp) <= .25 * INCH:
          print("previously tried grasp:", gr_action, gr_result)
          return (not gr_result)
    return False 

  def get_goal_states(self, policy):
    goal_states = []
    adjacent_clusters = []
    # results of comparing Goal to World State
    print("##############")
    print("Compare Goal to World")
    [self.goal_clusters.w_best_match, self.goal_clusters.w_unmatched, 
     self.goal_clusters.w_unmoved_match, self.goal_clusters.w_slightly_moved_match,
     self.goal_clusters.w_split, self.goal_clusters.w_poss_base_tuples,
     self.goal_clusters.w_combined_octobase, 
     self.goal_clusters.g_best_match, self.goal_clusters.g_unmatched, 
     self.goal_clusters.g_unmoved_match, self.goal_clusters.g_slightly_moved_match, 
     self.goal_clusters.g_split, self.goal_clusters.g_poss_base_tuples, 
     self.goal_clusters.obb_w_g_pct_ovlp, self.goal_clusters.obb_g_w_pct_ovlp] = policy.world_clusters.compare_clusters_to_world(self.goal_clusters)
    print("Compare Done")
    print("##############")

    policy.world_clusters.octobase = self.goal_clusters.w_combined_octobase, 
    for g_c_id,g_c in enumerate(self.goal_clusters.clusters):
      status = self.goal_clusters.clusters[g_c_id].cluster['status']
      if status != "BASE" and status != "EMPTY":
        del_from_shape = []
        for pt_i,pt in enumerate(self.goal_clusters.clusters[g_c_id].cluster['shape']):
          if pt in self.goal_clusters.g_poss_base_tuples:
            del_from_shape.append(pt_i)
        if len(del_from_shape) > 0:
          print(len(del_from_shape), " points in goal_cluster", g_c_id," are base")
        for pt_i in del_from_shape:
          # del from high to low position
          del self.goal_clusters.clusters[g_c_id].cluster['shape'][pt_i]

    self.goal_clusters.g_combined_octobase = self.goal_clusters.g_poss_base_tuples
    # [w_c_id, g_c_id, dist, w_loc, g_loc, [action]]
    for g_c_id,g_c in enumerate(self.goal_clusters.clusters):
      status = self.goal_clusters.clusters[g_c_id].cluster['status']
      if status != "ACTIVE":
        continue
      if self.goal_clusters.g_best_match[g_c_id] is None:
        if g_c_id in self.goal_clusters.g_unmatched:
          # g_c_id to world cluster mapping doesn't exist. 
          # Check if world cluster to g_c_id mapping exists. 
          if g_c_id in self.goal_clusters.w_best_match:
            for w_c_id2, g_c_id2 in enumerate(self.goal_clusters.w_best_match):
              if g_c_id2 == g_c_id:
                goal_state = [w_c_id2, g_c_id2, "unmatched"]
                goal_info = self.get_goal_info(policy, w_c_id2, g_c_id2)
                [dist, w_loc, g_loc, w_obb, g_obb] = goal_info
                goal_state.append(goal_info)
                goal_states.append(goal_state)
          else:
            # look for splits
            found = False
            for g_c_id2, g_splt in enumerate(self.goal_clusters.g_split):
              for w_c_id2 in self.goal_clusters.g_split[g_c_id2]:
                # if w_c_id2 == w_c_id:
                  print("pretty confident that ", w_c_id2, " is split of ", g_c_id2)
                  goal_state = [w_c_id2, g_c_id2, "split"]
                  goal_info = self.get_goal_info(policy, w_c_id2, g_c_id2)
                  [dist, w_loc, g_loc, w_obb, g_obb] = goal_info
                  goal_state.append(goal_info)
                  found = True
            [l_r, t_b, c_f] = [0,1,2]
            if not found:
              for w_c_id2, w_splt in enumerate(self.goal_clusters.w_split):
                for i, g_c_id2 in enumerate(self.goal_clusters.w_split[w_c_id2]):
                  # if w_c_id2 == w_c_id:
                  print("pretty confident that ", w_c_id2, " is split of ", g_c_id2)
                  if i == 0:
                    goal_state = [w_c_id2, g_c_id2, "combination"]
                    goal_info = self.get_goal_info(policy, w_c_id2, g_c_id2)
                    [dist0, w_loc0, g_loc0, w_obb0, g_obb0] = goal_info
                    goal_state.append(goal_info)
                    goal_states.append(goal_state)
                    w_dist[l_r][i] = distance_3d(w_obb0.points[0], w_obb0.points[1])
                    w_dist[t_b][i] = distance_3d(w_obb0.points[3], w_obb0.points[6])
                    w_dist[c_f][i] = distance_3d(w_obb0.points[1], w_obb0.points[2])
                  else:
                    goal_state = [w_c_id2, g_c_id2, "combination"]
                    goal_info = self.get_goal_info(policy, w_c_id2, g_c_id2)
                    [dist1, w_loc1, g_loc1, w_obb1, g_obb1] = goal_info
                    goal_state.append(goal_info)
                    goal_states.append(goal_state)
                    w_dist[l_r][i] = distance_3d(w_obb1.points[0], w_obb1.points[1])
                    w_dist[t_b][i] = distance_3d(w_obb1.points[3], w_obb1.points[6])
                    w_dist[c_f][i] = distance_3d(w_obb1.points[1], w_obb1.points[2])


                  # compare OBB height and see if there's stacking going on here
                  g_dist[l_r] = distance_3d(g_obb.points[0], g_obb.points[1])
                  g_dist[t_b] = distance_3d(g_obb.points[3], g_obb.points[6])
                  g_dist[c_f] = distance_3d(g_obb.points[1], g_obb.points[2])

                  # check permutations of obb orientations to see if fits
                  # hwd => height, width, depth  (3*2*1)
                  permutations = {{l_r, c_f, t_b},
                                  {l_r, t_b, c_f},
                                  {t_b, l_r, c_f},
                                  {t_b, c_f, l_r},
                                  {c_f, t_b, l_r},
                                  {c_f, l_r, t_b}}

                  for g in permutations:
                    for w0 in permutations:
                      for w1 in permutations:
		        if (abs(g_dist[g[0]] - (w_dist[w0[0]][0] + w_dist[w1[0]][1])) < .1*g_dist[g[0]]
		            and ((g_dist[g[1]] > w_dist[w0[1]][0] and
		                 abs(g_dist[g[1]] - w_dist[w1[1]][0]) < .5*abs(g_dist[g[1]]))
                              or
		                 (abs(g_dist[g[1]] - w_dist[w0[1]][0]) < .5*abs(g_dist[g[1]]))
		                  and (g_dist[g[1]] > w_dist[w1[1]][1]))
                            and ((g_dist[g[2]] > w_dist[w0[2]][0] and
                                 abs(g_dist[g[2]] - w_dist[w1[2]][0]) < .5*abs(g_dist[g[2]]))
                              or
                                 (abs(g_dist[g[2]] - w_dist[w0[2]][0]) < .5*abs(g_dist[g[2]])
                                  and (g_dist[g[2]] > w_dist[w1[2]][1])))):

                          # a nice snug fit found!
                          adjacent_clusters.append([g, w0, w1])
      elif g_c_id in self.goal_clusters.g_poss_base_tuples:
        continue
      else:
        w_c_id = self.goal_clusters.g_best_match[g_c_id]
        goal_state = [w_c_id, g_c_id, "best_match"]
        goal_info = self.get_goal_info(policy, w_c_id, g_c_id)
        [dist, w_loc, g_loc, w_obb, g_obb] = goal_info
        goal_state.append(goal_info)
        goal_states.append(goal_state)
      
    return goal_states, adjacent_clusters


  def centroid_push_start_end(self, gi_w_obb, gi_g_obb, gi_dist):
        x,y,z = 0,1,2
        # NUDGE centroids closer to each other
        # draw a line from w_centroid to g_centroid.
        # find point on w_obb on the line
        w_centroid = (gi_w_obb.centroid[x], gi_w_obb.centroid[y])
        g_centroid = (gi_g_obb.centroid[x], gi_g_obb.centroid[y])

        w_obb_x_min = gi_w_obb.points[0][x]
        w_obb_x_max = gi_w_obb.points[4][x]
        w_obb_y_min = gi_w_obb.points[0][y]
        w_obb_y_max = gi_w_obb.points[4][y]

        endpt = []
        endpt.append((w_obb_x_min, w_obb_y_min))
        endpt.append((w_obb_x_max, w_obb_y_min))
        endpt.append((w_obb_x_min, w_obb_y_max))
        endpt.append((w_obb_x_max, w_obb_y_max))

        combo = [[0,1],[1,4],[4,3],[3,0]]

        start_x,start_y = None, None
        for pt in combo:
          try:
            ln = (endpt[pt[0]], endpt[pt[1]])
            start_x,start_y = line_intersection((w_centroid, g_centroid), ln)
          except:
            continue
          if (w_obb_x_max >= start_x >= w_obb_x_min
              and w_obb_y_max >= start_y >= w_obb_y_min):
            break
        if start_x == None or start_y == None:
          return None, None
        line_seg = ((start_x,start_y),(w_centroid[x],w_centroid[y]))
        end_x, end_y = pt_on_line_seg(line_seg, gi_dist)
        return (start_x,start_y,gi_w_obb.centroid[z]), (end_x,end_y,gi_w_obb.centroid[z])

  def compute_theta(self, pt0, pt1, horiz = True):
    xdiff = (pt0[0] - pt1[0])
    ydiff = (pt0[1] - pt1[1])
    slope = ydiff / xdiff
    if horiz == True:
      theta = math.atan(slope)             # slope angle in radians
    elif math.atan(slope) < np.pi / 2:
      theta = math.atan(slope) + np.pi / 2
    else:
      theta = math.atan(slope) - np.pi / 2
    return theta

  def compute_gripper_width(self,w_c_id, policy):
    l,w = policy.world_clusters.clusters[w_c_id].obb_length_width()
    # GRIPPER_OPEN = [0.027, 0.027]
    gw  = min(.75*w, GRIPPER_WIDTH)
    gws = gw / GRIPPER_WIDTH * GRIPPER_OPEN[0]
    return [gw, gw]

  def goal_plan(self, policy, grasps, confidences, goal_states, gs_adjacent_clusters):
    def w_g_dist(gs):
      return gs[3][0]

    [x,y,z] = [0,1,2]
    self.goal_plan = sorted(goal_states, key=w_g_dist)

    # FUTURE: [FLIP_90,  [[x,y,z,theta][end theta]]]
    for gs_id,gs in enumerate(self.goal_plan):
      [gs_w_cid, gs_g_cid, gs_desc, gs_info] = gs
      [gi_dist, gi_w_loc, gi_g_loc, gi_w_obb, gi_g_obb] = gs_info

      # g_c = self.goal_clusters.clusters[gs_g_cid]
      g_c = self.goal_clusters
      w_g_pct_ovlp = g_c.obb_w_g_pct_ovlp[gs_w_cid][gs_g_cid] 
      g_w_pct_ovlp = g_c.obb_g_w_pct_ovlp[gs_g_cid][gs_w_cid]
      if (w_g_pct_ovlp != None and w_g_pct_ovlp > .7 and
          g_w_pct_ovlp != None and g_w_pct_ovlp > .7):
          print("pct_ovlp: ", g_c.obb_w_g_pct_ovlp, g_c.obb_g_w_pct_ovlp)
          # both the same size. Mostly same. Just rotate around.
          # future: compute obb rotations, for now see if this works
          # obb_height = gi_w_obb.points[5][z] + .1*INCH
          obb_height = gi_w_obb.points[0][z] + .1*INCH
          top_ctr_pt = [gi_w_obb.centroid[x],gi_w_obb.centroid[y], obb_height]
          theta = self.compute_theta(top_ctr_pt,[0,0,top_ctr_pt[z]])
          gw = self.compute_gripper_width(gs_w_cid, policy)
          action = [gs_w_cid, "ROTATE", top_ctr_pt, theta, gw, DEG20_IN_RADIANS]
          # future: compare to previous rotate and see how overlap changed
      elif (w_g_pct_ovlp != None and w_g_pct_ovlp > .3 and
            g_w_pct_ovlp != None and g_w_pct_ovlp > .3):
          start_pt, end_pt = self.centroid_push_start_end(gi_w_obb, gi_g_obb, gi_dist)
          theta = self.compute_theta(start_pt,end_pt,horiz=True)
          gw = self.compute_gripper_width(gs_w_cid, policy)
          action = [gs_w_cid, "NUDGE", start_pt, end_pt, theta, gw]
      elif policy.world_clusters.clusters[gs_w_cid].near_side() and not self.goal_clusters.clusters[gs_g_cid].near_side():
        ret = policy.world_clusters.clusters[gs_w_cid].plan_move_from_side()
        if ret == None:
          # why if near side????
          continue
        [obb_pt, side, dist, pt0, pt1, pt2] = ret
        # need to composate for half width of gripper for pt0, pt1?
        # side helps determine gripper open/close/theta
        if pt0 != None:
          theta = self.compute_theta(pt0,pt1)
        else:
          theta = self.compute_theta(pt1,pt2)
        gw = self.compute_gripper_width(gs_w_cid, policy)
        action = [gs_w_cid, "PUSH_FROM_EDGE", side, pt0, pt1, pt2, theta, gw]

      else:
        # Look for grasps not tried before. Try once per cluster.
        # try picking up w cluster and placing it to mapped location
        found = False
        cluster_grasps = []
        sum_conf = 0
        for grasp_id, [grasp, confidence] in enumerate(grasps):
          cluster_id = policy.world_clusters.cluster_contains(grasp)
          if cluster_id != gs_w_cid:
            continue
          # if policy.world_clusters.clusters[gs_w_cid].previously_tried_grasp(grasp):
          if self.previously_tried_grasp(gs_w_cid, grasp):
            # previously failed at a similar grasp
            print("previously failed grasp:",gs_w_cid)
            continue
          cluster_grasps.append(grasp)
          cluster_conf.append(confidences[grasp_id])
          sum_conf += confidences[grasp_id]
        if sum_conf > 0:
          cluster_conf /= sum_conf
          selected = np.random.choice(cluster_grasps, p = cluster_conf)
          action = [gs_w_cid, "PICK_PLACE", grasp, gi_g_obb.centroid]
          found = True

        # try pushing
        if not found:
          start_pt, end_pt = self.centroid_push_start_end(gi_w_obb, gi_g_obb, gi_dist)
          theta = self.compute_theta(start_pt,end_pt,horiz=True)
          gw = self.compute_gripper_width(gs_w_cid, policy)
          action = [gs_w_cid, "PUSH", start_pt, end_pt, theta, gw]
      self.goal_plan[gs_id].append(action)
    return self.goal_plan
 
  def goal_plan_moves(self, policy, grasps, confidences):
    goal_state, gs_adjacent_clusters = self.get_goal_states(policy)
    goal_moves = self.goal_plan(policy, grasps, confidences, goal_state, gs_adjacent_clusters)
    return goal_moves

# TODO:
# look for unexpect w_c_id movement for rolling
# look at estimate vs actual movement
# maybe round or cylindrical if much farther movement
# if obb height/width reversed, try flipping
# handle stacking
# self.cluster_move_history[c_id].append(["Grasp", grasp, False])

