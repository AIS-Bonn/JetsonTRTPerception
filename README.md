# JetsonTRTPerception
In this repo we publish the inference code running on the Jetson NX Smart Edge Sensors for the paper:<br>
3D Semantic Scene Perception using Distributed Smart Edge Sensors

## Citation
TODO (under review)

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

To cross-compile for jetson use instead:
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=10.2 -DCUDNN_LIB=/pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so -DCUBLAS_LIB=/usr/local/cuda-10.2/targets/aarch64-linux/lib/stubs/libcublas.so -DCUBLASLT_LIB=/usr/local/cuda-10.2/targets/aarch64-linux/lib/stubs/libcublasLt.so

make -j$(nproc)
```

Build the ros packages:<br>
```
cd ..
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```
