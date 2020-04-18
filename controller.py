#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PointStamped,
    Point,
    Quaternion,
    PoseStamped,
    Pose,
)
from moveit_commander import *
from moveit_msgs.srv import *
from arbotix_msgs.srv import *
from moveit_msgs.msg import *

from moveit_commander.exception import MoveItCommanderException

import numpy as np
import pickle
import traceback

from config import *


class WidowX:

    def __init__(self, boundaries=False):
        self.scene = PlanningSceneInterface()
        self.commander = MoveGroupCommander("widowx_arm")
        self.gripper = MoveGroupCommander("widowx_gripper")

        self.commander.set_end_effector_link('gripper_rail_link')

        if boundaries:
            self.add_bounds()

        self.joint_state_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.joint_callback)
        self.joint_pubs = [rospy.Publisher(
            '/%s/command' % name, Float64, queue_size=1) for name in JOINT_NAMES]
        self.gripper_pub = rospy.Publisher(
            '/gripper_prismatic_joint/command', Float64, queue_size=1)

        rospy.sleep(2)

    def joint_callback(self, joint_state):
        self.joint_state = joint_state

    def open_gripper(self, drop=False):
        plan = self.gripper_plan(GRIPPER_DROP if drop else GRIPPER_OPEN)
        return self.gripper.execute(plan, wait=True)

    def add_bounds(self):
        floor = PoseStamped()
        floor.header.frame_id = self.commander.get_planning_frame()
        floor.pose.position.x = 0
        floor.pose.position.y = 0
        floor.pose.position.z = .5
        self.scene.add_box('floor', floor, (1., 1., .001))

        leftWall2 = PoseStamped()
        leftWall2.header.frame_id = self.commander.get_planning_frame()
        leftWall2.pose.position.x = .2
        leftWall2.pose.position.y = 0
        leftWall2.pose.position.z = .475
        self.scene.add_box('leftWall2', leftWall2, (.001, .8, 1))

        leftWall = PoseStamped()
        leftWall.header.frame_id = self.commander.get_planning_frame()
        leftWall.pose.position.x = .18
        leftWall.pose.position.y = 0
        leftWall.pose.position.z = .475
        # self.scene.add_box('leftWall', leftWall, (.001, .35, .08))

        rightWall = PoseStamped()
        rightWall.header.frame_id = self.commander.get_planning_frame()
        rightWall.pose.position.x = -.18
        rightWall.pose.position.y = 0
        rightWall.pose.position.z = .475
        # self.scene.add_box('rightWall', rightWall, (.001, .35, .08))

        rightWall2 = PoseStamped()
        rightWall2.header.frame_id = self.commander.get_planning_frame()
        rightWall2.pose.position.x = -.2
        rightWall2.pose.position.y = 0
        rightWall2.pose.position.z = .475
        self.scene.add_box('rightWall2', rightWall2, (.001, .8, 1))

        frontWall = PoseStamped()
        frontWall.header.frame_id = self.commander.get_planning_frame()
        frontWall.pose.position.x = 0
        frontWall.pose.position.y = -.21
        frontWall.pose.position.z = .475
        self.scene.add_box('frontWall', frontWall, (.35, .001, .08))

        frontWall2 = PoseStamped()
        frontWall2.header.frame_id = self.commander.get_planning_frame()
        frontWall2.pose.position.x = 0
        frontWall2.pose.position.y = -.23
        frontWall2.pose.position.z = .475
        self.scene.add_box('frontWall2', frontWall2, (1, .001, 1))

        backWall = PoseStamped()
        backWall.header.frame_id = self.commander.get_planning_frame()
        backWall.pose.position.x = 0
        backWall.pose.position.y = .2
        backWall.pose.position.z = .475
        self.scene.add_box('backWall', backWall, (.35, .001, .08))

    def remove_bounds(self):
        for obj in self.scene.get_objects().keys():
            self.scene.remove_world_object(obj)

    def gripper_plan(self, joint_values):
        (success, plan_msg, planning_time, err_code) = self.gripper.plan(joint_values)
        print("success = ", success, " planning time = ", planning_time, " err code = ", err_code)
        return plan_msg

    def arm_plan(self, joint_values):
        (success, plan_msg, planning_time, err_code) = self.commander.plan(joint_values)
        print("success = ", success, " planning time = ", planning_time, " err code = ", err_code)
        # print(plan_msg)
        return plan_msg

    def close_gripper(self):
        plan = self.gripper_plan(GRIPPER_CLOSED)
        return self.gripper.execute(plan, wait=True)

    def get_ik_client(self, request):
        rospy.wait_for_service('/compute_ik')
        inverse_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        ret = inverse_ik(request)
        if ret.error_code.val != 1:
            return None
        return ret.solution.joint_state

    def get_fk_client(self, header, link_names, robot_state):
        rospy.wait_for_service('/compute_fk')
        fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        ret = fk(header, link_names, robot_state)
        if ret.error_code.val != 1:
            return None
        return ret.pose_stamped

    # def eval_grasp(self, threshold=.0001, manual=False):
    # unsuccessful grasp error: 0.00025979120707581276
    # sample successful grasp error: 0.013356082830967352
    def eval_grasp(self, threshold=.0003, manual=False):
        # print("eval_grasp: ", threshold, manual)
        if manual:
            user_input = None
            while user_input not in ('y', 'n', 'r'):
                user_input = raw_input(
                    'Successful grasp? [(y)es/(n)o/(r)edo]: ')
            if user_input == 'y':
                return 1, None
            elif user_input == 'n':
                return 0, None
            else:
                return -1, None
        else:
            current = np.array(self.gripper.get_current_joint_values())
            target = np.array(GRIPPER_CLOSED)
            error = current[0] - target[0]
            return (error > threshold), error

    def orient_to_target(self, x=None, y=None, angle=None):
        current = self.get_joint_values()
        if x or y:
            angle = np.arctan2(y, x)
        if angle >= np.pi:
            angle = np.pi
        elif angle <= -np.pi:
            angle = -np.pi
        if len(current) == 0:
          current.append(angle)
        else:
          print("current[0]=", angle)
          current[0] = angle
        #
        # plan = self.arm_plan(current)
        try:
            plan = self.arm_plan(current)
        except MoveItCommanderException as e:
            print('Exception while planning')
            traceback.print_exc(e)
            return False
        return self.commander.execute(plan, wait=True)

    def wrist_rotate(self, angle):
        rotated_values = self.commander.get_current_joint_values()
        rotated_values[4] = angle - rotated_values[0]
        if rotated_values[4] > np.pi / 2:
            rotated_values[4] -= np.pi
        elif rotated_values[4] < -(np.pi / 2):
            rotated_values[4] += np.pi
        plan = self.arm_plan(rotated_values)
        return self.commander.execute(plan, wait=True)

    def get_joint_values(self):
        return self.commander.get_current_joint_values()

    def get_current_pose(self):
        return self.commander.get_current_pose()

    def move_to_neutral(self):
        print('Moving to neutral...')
        plan = self.arm_plan(NEUTRAL_VALUES)
        return self.commander.execute(plan, wait=True)

    def move_to_drop(self, angle=None):
        drop_positions = DROPPING_VALUES[:]
        if angle:
            drop_positions[0] = angle
        plan = self.arm_plan(drop_positions)
        return self.commander.execute(plan, wait=True)

    def flip_and_drop(self, angle=(np.pi/2)):
        # flip 90 degrees and drop
        current = self.get_joint_values()
        if len(current) == 0:
          return
        print("current[3]=", angle)
        current[3] -= angle
        if angle >= np.pi:
            angle -= np.pi
        elif angle <= -np.pi:
            angle += np.pi
        try:
            plan = self.arm_plan(current)
        except MoveItCommanderException as e:
            print('Exception while planning')

    def move_to_empty(self):
        plan = self.arm_plan(EMPTY_VALUES)
        return self.commander.execute(plan, wait=True)

    def move_to_reset(self):
        print('Moving to reset...')
        plan = self.arm_plan(RESET_VALUES)
        return self.commander.execute(plan, wait=True)

    def orient_to_pregrasp(self, x, y):
        angle = np.arctan2(y, x)
        return self.move_to_drop(angle)

    def move_to_grasp(self, x, y, z, angle, compensate_control_noise=True):
        if compensate_control_noise:
            x = (x - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
            y = (y - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
        
        current_p = self.commander.get_current_pose().pose
        p1 = Pose(position=Point(x=x, y=y, z=z), orientation=DOWN_ORIENTATION)
        plan, f = self.commander.compute_cartesian_path(
            [current_p, p1], 0.001, 0.0)

        joint_goal = list(plan.joint_trajectory.points[-1].positions)

        first_servo = joint_goal[0]

        joint_goal[4] = (angle - first_servo) % np.pi
        if joint_goal[4] > np.pi / 2:
            joint_goal[4] -= np.pi
        elif joint_goal[4] < -(np.pi / 2):
            joint_goal[4] += np.pi

        try:
            plan = self.arm_plan(joint_goal)
        except MoveItCommanderException as e:
            print('Exception while planning')
            traceback.print_exc(e)
            return False

        return self.commander.execute(plan, wait=True)

    def move_to_vertical(self, z, force_orientation=True, shift_factor=1.0):
        current_p = self.commander.get_current_pose().pose
        current_angle = self.get_joint_values()[4]
        orientation = current_p.orientation if force_orientation else None
        p1 = Pose(position=Point(x=current_p.position.x * shift_factor,
                                 y=current_p.position.y * shift_factor, z=z), orientation=orientation)
        waypoints = [current_p, p1]
        plan, f = self.commander.compute_cartesian_path(waypoints, 0.001, 0.0)

        if not force_orientation:
            return self.commander.execute(plan, wait=True)
        else:
            if len(plan.joint_trajectory.points) > 0:
                joint_goal = list(plan.joint_trajectory.points[-1].positions)
            else:
                return False

            joint_goal[4] = current_angle

            plan = self.arm_plan(joint_goal)
            return self.commander.execute(plan, wait=True)

    def move_to_target(self, target):
        assert len(target) >= 6, 'Invalid target command'
        for i, pos in enumerate(target):
            self.joint_pubs[i].publish(pos)

    def move_to_joint_position(self, joints):
        """
        Adds the given joint values to the current joint values, moves to position
        """
        joint_state = self.joint_state
        joint_dict = dict(zip(joint_state.name, joint_state.position))
        for i in range(len(JOINT_NAMES)):
            joint_dict[JOINT_NAMES[i]] += joints[i]
        joint_state = JointState()
        joint_state.name = JOINT_NAMES
        joint_goal = [joint_dict[joint] for joint in JOINT_NAMES]
        joint_goal = np.clip(np.array(joint_goal), JOINT_MIN, JOINT_MAX)
        joint_state.position = joint_goal
        header = Header()
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        link_names = ['gripper_rail_link']
        position = self.get_fk_client(header, link_names, robot_state)
        target_p = position[0].pose.position
        x, y, z = target_p.x, target_p.y, target_p.z
        conditions = [
            x <= BOUNDS_LEFTWALL,
            x >= BOUNDS_RIGHTWALL,
            y <= BOUNDS_BACKWALL,
            y >= BOUNDS_FRONTWALL,
            z <= BOUNDS_FLOOR,
            z >= 0.15
        ]
        print("Target Position: %0.4f, %0.4f, %0.4f" % (x, y, z))
        for condition in conditions:
            if not condition:
                return
        self.move_to_target(joint_goal)
        rospy.sleep(0.15)

    def sweep_arena(self):
        self.remove_bounds()
        self.move_to_drop(.8)
        plan = self.arm_plan(TL_CORNER[0])
        self.commander.execute(plan, wait=True)

        plan = self.arm_plan(TL_CORNER[1])
        self.commander.execute(plan, wait=True)

        plan = self.arm_plan(L_SWEEP[0])
        self.commander.execute(plan, wait=True)

        plan = self.arm_plan(L_SWEEP[1])
        self.commander.execute(plan, wait=True)

        self.move_to_drop(-.8)
        plan = self.arm_plan(BL_CORNER[0])
        self.commander.execute(plan, wait=True)

        plan = self.arm_plan(BL_CORNER[1])
        self.commander.execute(plan, wait=True)

        self.move_to_drop(-2.45)
        plan = self.arm_plan(BR_CORNER[0])
        self.commander.execute(plan, wait=True)
        plan = self.arm_plan(BR_CORNER[1])
        self.commander.execute(plan, wait=True)

        self.move_to_drop(2.3)
        plan = self.arm_plan(TR_CORNER[0])
        self.commander.execute(plan, wait=True)
        plan = self.arm_plan(TR_CORNER[1])
        self.commander.execute(plan, wait=True)
        self.add_bounds()

    def discard_object(self):
        plan = self.arm_plan(PREDISCARD_VALUES)
        self.commander.execute(plan, wait=True)
        plan = self.arm_plan(DISCARD_VALUES)
        self.commander.execute(plan, wait=True)
        self.open_gripper(drop=True)
        plan = self.arm_plan(PREDISCARD_VALUES)
        self.commander.execute(plan, wait=True)
        self.move_to_neutral()


def main():
    rospy.init_node("widowx_custom_controller")
    widowx = WidowX()

    print('For debugging purposes')
    import pdb
    pdb.set_trace()
    pass

if __name__ == '__main__':
    main()
