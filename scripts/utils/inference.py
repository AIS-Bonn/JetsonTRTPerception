#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import threading

import utils.engine as engine_utils # TRT Engine creation/save/load utils
import utils.model as model_utils # UFF conversion uttils

import utils.common as common


# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_onnx_engine(onnx_file_path, engine_file_path, trt_engine_datatype, trt_logger, batch_size = 8):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(trt_logger) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, trt_logger) as parser, trt.Runtime(trt_logger) as runtime:
            config.max_workspace_size = 1 << 30 # 1024MiB
            if trt_engine_datatype == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)
            builder.max_batch_size = batch_size
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 8. Reshape input to given batch size
            network.get_input(0).shape = [batch_size, 3, 256, 128]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, pose=False, reid=False):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """
        self.cfx = cuda.Device(0).make_context()

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        if pose:
            print('\nPoseEngine: ')
        elif reid:
            print('\nReID-Engine: ')
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))
        
        if reid:
            self.trt_engine = get_onnx_engine(uff_model_path, trt_engine_path, trt_engine_datatype, TRT_LOGGER, batch_size)
        
        else:
            # If engine is not cached, we need to build it
            if not os.path.exists(trt_engine_path):
                # This function uses supplied .uff file
                # alongside with UffParser to build TensorRT
                # engine. For more details, check implmentation
                self.trt_engine = engine_utils.build_engine(
                    uff_model_path, TRT_LOGGER,
                    trt_engine_datatype=trt_engine_datatype,
                    batch_size=batch_size,
                    pose=pose)
                # Save the engine to file
                engine_utils.save_engine(self.trt_engine, trt_engine_path)

            # If we get here, the file with engine exists, so we can load it
            if not self.trt_engine:
                print("Loading cached TensorRT engine from {}".format(
                    trt_engine_path))
                self.trt_engine = engine_utils.load_engine(
                    self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine, explicit_batch = reid)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        if reid:
            input_volume = trt.volume([3, 256, 128])
        elif pose:
            input_volume = trt.volume(model_utils.ModelDataPose.INPUT_SHAPE)
        else:
            input_volume = trt.volume(model_utils.ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, img):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """
        threading.Thread.__init__(self)
        self.cfx.push()
        
        # Load image into CPU
        #print('\nInference: input image shape: {}, dtype: {}'.format(img.shape, img.dtype))

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel()) # will cast float64 to float32!

        # When infering on single image, we measure inference
        # time to output it to the user
        #inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keepCount_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        
        self.cfx.pop()
        
        # Output inference time
        #print("TensorRT inference time: {} ms".format(
            #int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keepCount_out
    
    def infer_batch_pose(self, crops_np = None):
        threading.Thread.__init__(self)
        self.cfx.push()
        # Verify if the supplied batch size is not too big
        if crops_np is None:
            crops_np = self.numpy_array
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = crops_np.shape[0]
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image crop list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))
        
        np.copyto(self.inputs[0].host, crops_np.ravel())
        
        #inference_start_time = time.time()

        # Fetch output from the model
        heatmaps = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream, batch_size=max_batch_size)
        
        self.cfx.pop()

        # Output inference time
        #print("\nPose: TensorRT inference time: {} ms".format(
            #int(round((time.time() - inference_start_time) * 1000))))
        
        return heatmaps

    def infer_batch_reid(self, crops_np = None):
        threading.Thread.__init__(self)
        self.cfx.push()
        # Verify if the supplied batch size is not too big
        if crops_np is None:
            crops_np = self.numpy_array
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = crops_np.shape[0]
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image crop list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))
        
        np.copyto(self.inputs[0].host, crops_np.ravel())
        
        #inference_start_time = time.time()

        # Fetch output from the model
        feat_reid = common.do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        
        self.cfx.pop()

        # Output inference time
        #print("\nPose: TensorRT inference time: {} ms".format(
            #int(round((time.time() - inference_start_time) * 1000))))
        
        return feat_reid
    
    def destory(self):
        self.cfx.pop()
