/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_GRID_ANCHOR_PLUGIN_H
#define TRT_GRID_ANCHOR_PLUGIN_H
#include "cudnn.h"
#include "kernel.h"
#include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class GridAnchorGenerator : public IPluginV2Ext
{
public:
    GridAnchorGenerator(const GridAnchorParameters* param, int numLayers, const char* version);

    GridAnchorGenerator(const void* data, size_t length, const char* version);

    ~GridAnchorGenerator() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

protected:
    std::string mPluginName;

private:
    Weights copyToDevice(const void* hostData, size_t count);

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;

    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    int mNumLayers;
    std::vector<GridAnchorParameters> mParam;
    int* mNumPriors;
    Weights *mDeviceWidths, *mDeviceHeights;
    std::string mPluginNamespace;
};

class GridAnchorDynamicGenerator : public IPluginV2DynamicExt
{
public:
    GridAnchorDynamicGenerator(const GridAnchorParameters* param, int numLayers, const char* version);
    GridAnchorDynamicGenerator(const void* data, size_t length, const char* version);
    ~GridAnchorDynamicGenerator() override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)  noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

protected:
    std::string mPluginName;

private:
    Weights copyToDevice(const void* hostData, size_t count);

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;

    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    int mNumLayers;
    std::vector<GridAnchorParameters> mParam;
    int* mNumPriors;
    Weights *mDeviceWidths, *mDeviceHeights;
    std::string mPluginNamespace;
};

class GridAnchorBasePluginCreator : public BaseCreator
{
public:
    GridAnchorBasePluginCreator();

    ~GridAnchorBasePluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

protected:
    std::string mPluginName;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

class GridAnchorPluginCreator : public GridAnchorBasePluginCreator
{
public:
    GridAnchorPluginCreator();
    ~GridAnchorPluginCreator() override = default;
};

class GridAnchorRectPluginCreator : public GridAnchorBasePluginCreator
{
public:
    GridAnchorRectPluginCreator();
    ~GridAnchorRectPluginCreator() override = default;
};

class GridAnchorDynamicBasePluginCreator : public BaseCreator
{
public:
    GridAnchorDynamicBasePluginCreator();
    ~GridAnchorDynamicBasePluginCreator() override = default;
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

protected:
    std::string mPluginName;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

class GridAnchorDynamicPluginCreator : public GridAnchorDynamicBasePluginCreator
{
public:
    GridAnchorDynamicPluginCreator();
    ~GridAnchorDynamicPluginCreator() override = default;
};

class GridAnchorRectDynamicPluginCreator : public GridAnchorDynamicBasePluginCreator
{
public:
    GridAnchorRectDynamicPluginCreator();
    ~GridAnchorRectDynamicPluginCreator() override = default;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GRID_ANCHOR_PLUGIN_H
