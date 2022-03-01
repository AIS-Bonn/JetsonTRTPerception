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

# This class contains converted (UFF) model metadata
class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (3, 480, 640)
    # Name of output node
    OUTPUT_NAME = "NMS"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]
    
# This class contains converted (UFF) model metadata
class ModelDataPose(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (3, 256, 192)
    # Name of output node
    OUTPUT_NAME = "tower_0/out/BiasAdd"

    @staticmethod
    def get_input_channels():
        return ModelDataPose.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelDataPose.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelDataPose.INPUT_SHAPE[2]
