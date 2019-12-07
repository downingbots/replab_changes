For now:
  just random drop

class world:

  def 
  object_list

  list of objects

class object

  def cluster_state():
    self.cluster = 
    self.cluster_location = (x, y, z)
    self.cluster_orientation =
    self.cluster_center = based on simple bounding box
  def cluster_attributes():
    self.cluster_grasps = grasps, locations, and (success, failure, unknown)
    self.cluster_keypoints = use for cluster identification, orientation, 
                             extension
    self.cluster_push = 
    self.cluster_flat   = use segmentation, look for top/bottom flat spots
    self.cluster_round  = Do later but see if roles in a direction
    self.cluster_concave  = Do later but ID holes > 1"?
    self.soft_hard if shape changes significantly, then soft
                   Just interact with other soft objects 
                   or if no more hard objects to play with
    self.rectangular_cylindar_ball_other =
    self.friction
    self.roll_direction
  def split_clusters():
    separate clusters into separate interacting objects
  def extend_cluster():
    separate clusters into separate interacting objects
  def actions():
    rotate 20 DEGREES (until 360)
      use keypoints to glue together clusters
    flip 180 DEGREES 
    flip 90 DEGREES 
    push
    stack
    random drop
  def prioritize_actions():
    identify objects
      - find clusters, keypoints, planes
    find object shape by rotation, flip
    find object attributes by push, drop
    find object interactions by combinatorics of objects & actions
        (rolling, stacking, dropping)
        prioritizing of objects (stacking of flat surfaces, 
          rolling of rollable objects, dropping of soft objects...)

by push, drop
class object_interaction:

  priority()
    cluster 
       rotate until shape understood
    understand_interaction_of_objects
       placed_inside (success/failure)
                     (combined object)
       stacked_on
        
    combination_of_interactions

class history:
   history_of_object_interaction
   
