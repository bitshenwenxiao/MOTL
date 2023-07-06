# MOTL
Vision based Multi-object Tracking and Localization by UAV Swarm

Hao Shen, Defu Lin, Xiwen Yang, Shaoming He 

The  code of our paper "Vision based Multi-object Tracking and Localization by UAV Swarm" (The link will be updated after publication).   The code is based on the [VehicleNet](https://github.com/michuanhaohao/AICITY2021_Track2_DMT).
The video of the constructed situation in AirSim simulation system  is available at [YouTube](https://youtu.be/erCiENAOEaM).

## test on MUMO dataset
### 1. download MUMO dataset
you should first download the [MUMO dataset](https://github.com/bitshenwenxiao/MUMO).
### 2. run object detection
you should run object detection on MUMO dataset. The results format is 

    x_left_top, y_left_top, x_right_bottom, y_right_bottom
We open our detection results in the compressed file "[detection.zip](https://drive.google.com/file/d/1PDd3DV9dstR08AyvOcSmbsy3nFcz62FH/view?usp=share_link)". For fair comparison, we suggest you use the same the directory detection results for different solutions of multi-UAV multi-object tracking.  If you run other object detectors, the directory organization of results should be same as ours.
### 3. run our solution for multi-UAV multi-object Tracking.
You should revise these directory paths at lines 116-118 in the file "main.py".

    path_dataset = '/'  #the directoy path of MUMO dataset
    path_det = 'result/detections/' #the directoy path of object detections
    path_out = './result/id/' #the results of the solution.
