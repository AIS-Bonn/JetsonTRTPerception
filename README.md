# JetsonTRTPerception
In this repo we publish the inference code running on the Jetson NX Smart Edge Sensors for the paper:<br>
3D Semantic Scene Perception using Distributed Smart Edge Sensors

## Citation
Simon Bultmann and Sven Behnke:<br>
3D Semantic Scene Perception using Distributed Smart Edge Sensors<br>
Accepted for 17th International Conference on Intelligent Autonomous Systems (IAS), Zagreb, Croatia, June 2022.

## Installation
### Dependencies
The code was tested with ROS melodic and Ubuntu 18.04.

On the host PC, TensorRT is required, install it via the official documentation.
The code was tested with:<br>
Host PC: TensorRT 7.2.3-1+cuda11.1<br>
Jetson NX: Jetpack 4.5.1 (TensorRT 7.1.3, CUDA 10.2)

The code is not currently compatible with TensorRT 8.x, as the UFF-Parser is used for RGB detection and pose estimation models.

We depend on a custom version of TensorRT Open Source Software (OSS), included in this repo. The `gridAnchor` and `flattenConcat` plugins have been updated to support dyncamic shapes, and the `gridAnchor` plugin has been fixed to support rectangular input images.

### ROS packages
Clone this repo and the `SmartEdgeSensor3DScenePerception` repo inside your catkin workspace:<br>
```
cd catkin_ws/src
git clone https://github.com/AIS-Bonn/JetsonTRTPerception.git
git clone https://github.com/AIS-Bonn/SmartEdgeSensor3DScenePerception.git
```

Build TensorRT OSS:<br>
```
cd JetsonTRTPerception/TensorRT
mkdir -p build && cd build

export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu/
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out 

make -j$(nproc)
```
For furhter build instructions (Native build on Jetson / cross-compile for Jetson) see:
https://github.com/NVIDIA/TensorRT#building-tensorrt-oss

Build the ros packages:<br>
```
cd ..
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

### Launch
`roslaunch jetson_trt_pose pose_estimation.launch feedback:=skel camera:=d455`

color image topic: `/d455/color/image_raw`<br>
color camera info: `/d455/color/camera_info`<br>
depth image topic: `/d455/depth/image_rect_raw`<br>
depth camera info: `/d455/depth/camera_info`<br>

feedback topic: `/d455/skel_pred`

If thermal detector model is given:<br>
thermal image topic: `/d455/lepton/image`<br>
thermal camera info: `/d455/lepton/camera_info`<br>
