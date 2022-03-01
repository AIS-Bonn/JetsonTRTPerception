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

# uff_ssd path management singleton class
import os
import sys
import tensorrt as trt


class Paths(object):
    def __init__(self):
        self._SAMPLE_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            os.pardir
        )
        self._WORKSPACE_DIR_PATH = self._SAMPLE_ROOT
        self._VOC_DIR_PATH = \
            os.path.join(self._SAMPLE_ROOT, 'VOCdevkit', 'VOC2007')

    # User configurable paths

    def set_workspace_dir_path(self, workspace_dir):
        self._WORKSPACE_DIR_PATH = workspace_dir

    def get_workspace_dir_path(self):
        return self._WORKSPACE_DIR_PATH

    # Fixed paths

    def get_sample_root(self):
        return self._SAMPLE_ROOT

    def get_models_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'models')

    def get_engines_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'engines')
    
    def get_engines_pose_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'engines_pose')

    def get_engine_path(self, inference_type=trt.DataType.FLOAT, max_batch_size=1):
        inference_type_to_str = {
            trt.DataType.FLOAT: 'FLOAT',
            trt.DataType.HALF: 'HALF',
            trt.DataType.INT32: 'INT32',
            trt.DataType.INT8: 'INT8'
        }
        return os.path.join(
            self.get_engines_dir_path(),
            inference_type_to_str[inference_type],
            'engine_bs_{}.buf'.format(max_batch_size))
    
    def get_engine_pose_path(self, inference_type=trt.DataType.FLOAT, max_batch_size=1):
        inference_type_to_str = {
            trt.DataType.FLOAT: 'FLOAT',
            trt.DataType.HALF: 'HALF',
            trt.DataType.INT32: 'INT32',
            trt.DataType.INT8: 'INT8'
        }
        return os.path.join(
            self.get_engines_pose_dir_path(),
            inference_type_to_str[inference_type],
            'engine_bs_{}.buf'.format(max_batch_size))

    def get_model_dir_path(self, model_name):
        return os.path.join(self.get_models_dir_path(), model_name)

    def get_model_pb_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_inference_graph.pb'
        )

    def get_model_uff_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_inference_graph.uff'
        )
    
    def get_model_pose_pb_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_graph.pb'
        )
    
    def get_model_pose_uff_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_graph.uff'
        )

    # Paths correctness verifier

    def verify_all_paths(self, should_verify_voc=False):
        error = False

        if not os.path.exists(self.get_workspace_dir_path()):
            error = True

        if error:
            print("An error occured when running the script.")
            sys.exit(1)

PATHS = Paths()
