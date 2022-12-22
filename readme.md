
Implementation of the paper “CNN-BASED OBJECT DETECTION AND DISTANCE PREDICTION FOR AUTONOMOUS DRIVING USING STEREO IMAGES”.</br>
# Weights
[imagenet pretrain (cls100)](https://drive.google.com/file/d/192eLCa2ADovci79V0M6QYmONv3Hxh9xp/view?usp=sharing)</br>
[sceneflow pretrain](https://drive.google.com/file/d/1LQhWm327ED-SWHg7SDCqjtzY8d1xYrsP/view?usp=sharing)</br>
[kitti 2d stereo](https://drive.google.com/file/d/1pKywmO6sLGbtf0A0C8h3_Stt-RN-wSVa/view?usp=sharing)</br>

# ...
It describes how to pretrain a stereoneck and train the entire network end-to-end.</br>
It will run without problems in the latest version of PyTorch.</br>
The version at the time we tested is 1.10.0.</br>
Modify the paths inside the CFG file appropriately.</br>
The 2D bounding box and distance labels are enclosed in the gendata folder. Unzip it.</br>
Code to convert the 3D bounding box of the KITTI form to a 2D bounding box and distance is included in result_fromkitti.py.</br>
The code to output the 2D bounding box and distance predicted by the network to a txt file is included in trainkitti2d_.py.</br>

# Samples
![CIASSD](https://github.com/sjg918/kitti-2d-stereo/blob/main/result/CIASSD.png?raw=true)
Result using pseudo LiDAR points generated from disp map of PSMNet: CIA-SSD was used for 3Ddet network.</br>
[CIA-SSD](https://github.com/Vegeta2020/CIA-SSD)</br>
[PSMNet](https://github.com/JiaRenChang/PSMNet)</br>

![StereoRCNN](https://github.com/sjg918/kitti-2d-stereo/blob/main/result/StereoRCNN.png?raw=true)
[Stereo-RCNN](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/tree/1.0)</br>

![YOLOStereo3D](https://github.com/sjg918/kitti-2d-stereo/blob/main/result/YOLOStereo3D.png?raw=true)
[visualdet3D](https://github.com/Owen-Liuyuxuan/visualDet3D)</br>

![ours](https://github.com/sjg918/kitti-2d-stereo/blob/main/result/ours.png?raw=true)
This is our output.</br>

# special thanks
[YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection](https://arxiv.org/abs/2103.09422)</br>
[Improved Stereo Matching Accuracy Based on Selective Backpropagation and Extended Cost Volume](https://link.springer.com/article/10.1007/s12555-021-0724-6)</br>
