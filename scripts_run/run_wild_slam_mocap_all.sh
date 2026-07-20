#!/bin/bash
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/person_tracking.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/stones.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/table_tracking2.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/table_tracking1.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/racket.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ball.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/umbrella.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ANYmal1.yaml
sleep 60
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/ANYmal2.yaml
