#!/bin/bash
python run.py  ./configs/Dynamic/Bonn/bonn_crowd2.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_crowd.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_balloon2.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_person_tracking.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_person_tracking2.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_moving_nonobstructing_box.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2.yaml
sleep 60
python run.py  ./configs/Dynamic/Bonn/bonn_balloon.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/table_tracking2.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/person_tracking.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/stones.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/table_tracking2.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/table_tracking1.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/racket.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ball.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/umbrella.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ANYmal1.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ANYmal2.yaml
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg2_desk_with_person.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_static.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_halfsphere_static.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_halfsphere.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml
python run.py  ./configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_static.yaml

