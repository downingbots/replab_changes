#!/usr/bin/env python

import numpy as np
import rospy
import time
import argparse
from executor import *
from pickpushpolicy import *
from policy import *
from replab_core.config import *
from replab_core.utils import *
from numpy.random import choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000,
                        help="Number of samples to collect")
    parser.add_argument('--datapath', type=str,
                        default='', help="Path for saving data samples")
    parser.add_argument('--save', type=int, default=0,
                        help="Toggles whether samples are saved")
    parser.add_argument('--start', type=int, default=0,
                        help="Starting index for sample numbering")
    parser.add_argument('--method', type=str, default='pickpush',
                        help="Method used for planning grasps", choices=METHODS)
    parser.add_argument('--calibrate', type=str, default='none',
                        help="Method used for calibrating grasps", 
                        choices=CALIBRATION_METHODS)
    parser.add_argument('--email', action="store_true", default=False,
                        help="Send an email if data collection is interrupted (email settings must be configured correctly)")

    # print(METHODS)
    args = parser.parse_args()

    assert args.method in METHODS

    try:
        rospy.init_node("executor_widowx")
        executor = Executor(scan=False, datapath=args.datapath, save=args.save)

        executor.widowx.move_to_neutral()
        executor.widowx.open_gripper()

        counter = 0

        rospy.sleep(1)

        start = time.time()

        if args.method == 'principal-axis':
            policy = PrincipalAxis()
        elif args.method == 'pinto2016':
            policy = Pinto2016(PINTO2016_PRETRAINED_WEIGHTS)
        elif args.method == 'datacollection':
            policy = DataCollection(noise=True)
        elif args.method == 'datacollection-noiseless':
            policy = DataCollection(noise=False)
        elif args.method == 'fullimage':
            policy = FullImage(FULLIMAGE_PRETRAINED_WEIGHTS)
        elif args.method == 'combined':
            policy_array = [DataCollection(noise=True), PrincipalAxis()]
            policy_array_weights = [0.67, 0.33]
            policy = choice(policy_array, 1, policy_array_weights)[-1]
        elif args.method == 'pickpush':
            policy = PickPushCollection()
        else:
            print('Method not recognized, exiting')
            exit()

        if args.calibrate == 'manual':
            executor.set_calibration(True)

        sample_id = args.start
        while sample_id < args.start + args.samples:

            print('\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n')
            print('Grasp %d' % sample_id)

            pc = executor.get_pc()
            if args.method != 'pickpush':
              rgbd = executor.get_rgbd()
            else:
              # in pickpush, get_pc publishes to the octomap server and
              # get_octomap_pc returns the parameters to plan_graps
              octomap, header = executor.get_octomap_pc()

            if len(pc) == 0:
              print("pc len 0")
              rospy.sleep(1)
              continue
            executor.sample['filtered_pc'] = pc

            try:
                if args.method == 'combined':
                    policy = choice(policy_array, 1, policy_array_weights)[-1]
                if args.method != 'pickpush':
                  grasps = policy.plan_grasp(rgbd, pc)
                else:
                  grasps = policy.plan_grasp(octomap, header)
                if grasps == None:
                  continue
            except ValueError as ve:
                traceback.print_exc(ve)
                print('Error planning, resetting...')
                executor.widowx.move_to_neutral()
                executor.widowx.open_gripper()
                continue

            if grasps == None or len(grasps) == 0:
                print('No grasps plannable, sweeping rig')
                executor.widowx.sweep_arena()
                executor.widowx.move_to_neutral()
                executor.widowx.open_gripper()
                continue

            confidences = []
            kept_indices = []
            calib_grasps = []

            for i, (grasp, confidence) in enumerate(grasps):
                print(i," grasp:", grasp)
                if inside_polygon([grasp[0],grasp[1],grasp[2]], END_EFFECTOR_BOUNDS):
                    kept_indices.append(i)
                    confidences.append(confidence)
                    calib_grasps.append(grasp)
                else:
                   print("outside_polygon:", [grasp[0],grasp[1],grasp[2]])
            # print("calib_grasps", calib_grasps)

            if len(confidences) == 0:
                print('All planned grasps out of bounds / invalid, resetting...')
                executor.widowx.move_to_neutral()
                executor.widowx.open_gripper()
                continue

            # print("confidences", confidences)
            selected = np.random.choice(np.argsort(confidences)[-5:])
            grasp = grasps[kept_indices[selected]][0]

            if args.calibrate == 'manual':
              print("Calibrate grasps")
              executor.sample['calibrate'] = 'manual'
              executor.publish_grasps(grasps, calib_grasps[0])
              executor.record_grasp(calib_grasps[0], grasps)
              success, err = executor.calibrate_grasp(calib_grasps, confidences,  policy)
            elif args.method == 'pickpush':
              print("Publish grasps")
              executor.publish_grasps(grasps, grasp)
              executor.record_grasp(grasp, grasps)
              success, err = executor.execute_grasp(grasp, grasps, confidences,  policy)
            else:
              success, err = executor.execute_grasp(grasp)

            if err:
                executor.widowx.move_to_reset()
                executor.widowx.open_gripper(drop=True)
                executor.widowx.move_to_neutral()
                executor.widowx.open_gripper()
                continue

            if args.save:
                executor.save_sample(sample_id)

            print('Success: %r %f' % (success, err))

            if counter % 500 == 499:
                executor.widowx.sweep_arena()
                executor.widowx.move_to_neutral()
            sample_id += 1
            counter += 1
            rospy.sleep(1)

        end = time.time()
        print('Time elapsed : %.2f' % (end - start))

    except Exception as e:
        traceback.print_exc(e)
        print('Exception encountered, terminating')

    print('Done')

if __name__ == '__main__':
    main()
