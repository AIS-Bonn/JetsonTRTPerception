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

#include "flattenConcat.h"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::FlattenConcat;
using nvinfer1::plugin::FlattenConcatPluginCreator;
using nvinfer1::plugin::FlattenConcatDynamic;
using nvinfer1::plugin::FlattenConcatDynamicPluginCreator;

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAMES[] = {"FlattenConcat_TRT", "FlattenConcatDynamic_TRT"};
}

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

PluginFieldCollection FlattenConcatDynamicPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatDynamicPluginCreator::mPluginAttributes;

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcatDynamic::FlattenConcatDynamic(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis,
    const int* inputConcatAxis, const size_t* copySize)
    : mCopySize(numInputs)
    , mInputConcatAxis(numInputs)
    , mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
    , mOutputConcatAxis(outputConcatAxis)
    , mNumInputs(numInputs)
{
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);

    std::copy(copySize, copySize + mNumInputs, mCopySize.begin());
    std::copy(inputConcatAxis, inputConcatAxis + mNumInputs, mInputConcatAxis.begin());
}

FlattenConcatDynamic::FlattenConcatDynamic(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis,
    const int* inputConcatAxis, const size_t* copySize)
    : mCopySize(numInputs)
    , mInputConcatAxis(numInputs)
    , mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
    , mOutputConcatAxis(outputConcatAxis)
    , mNumInputs(numInputs)
{
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);

    std::copy(copySize, copySize + mNumInputs, mCopySize.begin());
    std::copy(inputConcatAxis, inputConcatAxis + mNumInputs, mInputConcatAxis.begin());
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data);
    const char* const a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);

    mInputConcatAxis.resize(mNumInputs);
    std::for_each(mInputConcatAxis.begin(), mInputConcatAxis.end(), [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    mCopySize.resize(mNumInputs);
    std::for_each(mCopySize.begin(), mCopySize.end(), [&](size_t& inp) { inp = read<size_t>(d); });

    ASSERT(d == a + length);
}

FlattenConcatDynamic::FlattenConcatDynamic(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data);
    const char* const a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);

    mInputConcatAxis.resize(mNumInputs);
    std::for_each(mInputConcatAxis.begin(), mInputConcatAxis.end(), [&](int& inp) { inp = read<int>(d); });

    mNCHW = read<nvinfer1::Dims4>(d);

    mCopySize.resize(mNumInputs);
    std::for_each(mCopySize.begin(), mCopySize.end(), [&](size_t& inp) { inp = read<size_t>(d); });

    ASSERT(d == a + length);
}

FlattenConcat::~FlattenConcat() {}
FlattenConcatDynamic::~FlattenConcatDynamic() {}

int FlattenConcat::getNbOutputs() const
{
    return 1;
}

int FlattenConcatDynamic::getNbOutputs() const noexcept
{
    return 1;
}

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims >= 1);
    ASSERT(index == 0);

    mNumInputs = nbInputDims;
    mCopySize.resize(mNumInputs);
    mInputConcatAxis.resize(mNumInputs);
    int outputConcatAxis = 0;

    for (int i = 0; i < nbInputDims; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[2] == inputs[0].d[2]);
        }
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        outputConcatAxis += flattenInput;
    }

    return DimsCHW(mConcatAxisID == 1 ? outputConcatAxis : 1, mConcatAxisID == 2 ? outputConcatAxis : 1,
        mConcatAxisID == 3 ? outputConcatAxis : 1);
}

DimsExprs FlattenConcatDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    ASSERT(nbInputs >= 1);
    ASSERT(outputIndex == 0);
    
    mNumInputs = nbInputs;
    mCopySize.resize(mNumInputs);
    mInputConcatAxis.resize(mNumInputs);
    int outputConcatAxis = 0;
    
    ASSERT(mConcatAxisID >= 1 && mConcatAxisID <= 3);
    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 4);
        ASSERT(inputs[i].d[1]->isConstant() && inputs[i].d[2]->isConstant() && inputs[i].d[3]->isConstant());
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[1]->getConstantValue() == inputs[0].d[1]->getConstantValue());
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[2]->getConstantValue() == inputs[0].d[2]->getConstantValue());
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[3]->getConstantValue() == inputs[0].d[3]->getConstantValue());
        }
        flattenInput = inputs[i].d[1]->getConstantValue() * inputs[i].d[2]->getConstantValue() * inputs[i].d[3]->getConstantValue();
        outputConcatAxis += flattenInput;
    }
    
    // Output dimensions
    // index 0 : Dimensions 1x param.keepTopK x 7
    // index 1: Dimensions 1x1x1
    DimsExprs out_dim;
    out_dim.nbDims = 4;
    out_dim.d[0] = inputs[0].d[0];
    out_dim.d[1] = mConcatAxisID == 1 ? exprBuilder.constant(outputConcatAxis): exprBuilder.constant(1);
    out_dim.d[2] = mConcatAxisID == 2 ? exprBuilder.constant(outputConcatAxis): exprBuilder.constant(1);
    out_dim.d[3] = mConcatAxisID == 3 ? exprBuilder.constant(outputConcatAxis): exprBuilder.constant(1);
    return out_dim;
    
}

int FlattenConcat::initialize()
{
    return STATUS_SUCCESS;
}

void FlattenConcat::terminate() {}

size_t FlattenConcat::getWorkspaceSize(int) const
{
    return 0;
}

int FlattenConcatDynamic::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void FlattenConcatDynamic::terminate() noexcept {}

size_t FlattenConcatDynamic::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int FlattenConcat::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    ASSERT(mConcatAxisID != 0);
    // mCHW is the first input tensor
    int numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

    // Num concats will be proportional to number of samples in a batch
    if (!mIgnoreBatch)
    {
        numConcats *= batchSize;
    }

    auto* output = static_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = static_cast<const float*>(inputs[i]);
        for (int n = 0; n < numConcats; ++n)
        {
            CUBLASASSERT(cublasScopy(mCublas, mInputConcatAxis[i], input + n * mInputConcatAxis[i], 1,
                output + (n * mOutputConcatAxis + offset), 1));
        }
        offset += mInputConcatAxis[i];
    }

    return STATUS_SUCCESS;
}

int FlattenConcatDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    ASSERT(mConcatAxisID != 0);
    // mCHW is the first input tensor
    int numConcats = std::accumulate(mNCHW.d, mNCHW.d + mConcatAxisID, 1, std::multiplies<int>());

    // Num concats will be proportional to number of samples in a batch
//     if (!mIgnoreBatch) -> Batch is accounted for in expclicit batch dimension N
//     {
//         numConcats *= batchSize;
//     }

    auto* output = static_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = static_cast<const float*>(inputs[i]);
        for (int n = 0; n < numConcats; ++n)
        {
            CUBLASASSERT(cublasScopy(mCublas, mInputConcatAxis[i], input + n * mInputConcatAxis[i], 1,
                output + (n * mOutputConcatAxis + offset), 1));
        }
        offset += mInputConcatAxis[i];
    }

    return STATUS_SUCCESS;
}

size_t FlattenConcat::getSerializationSize() const
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims)
        + (sizeof(decltype(mCopySize)::value_type) * mNumInputs);
}

size_t FlattenConcatDynamic::getSerializationSize() const noexcept
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims)
        + (sizeof(decltype(mCopySize)::value_type) * mNumInputs);
}

void FlattenConcat::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* const a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

void FlattenConcatDynamic::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* const a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mNCHW);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void FlattenConcat::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
    mCublas = cublasContext;
}

void FlattenConcatDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
    mCublas = cublasContext;
}

// Detach the plugin object from its execution context.
void FlattenConcat::detachFromContext() {}
void FlattenConcatDynamic::detachFromContext() {}

// Return true if output tensor is broadcast across a batch.
bool FlattenConcat::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool FlattenConcat::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Set plugin namespace
void FlattenConcat::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

void FlattenConcatDynamic::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* FlattenConcat::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

const char* FlattenConcatDynamic::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType FlattenConcat::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index < 3); // One Output ?!
    return DataType::kFLOAT;
}

DataType FlattenConcatDynamic::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(index == 0); // One Output ?!
    return DataType::kFLOAT;
}

void FlattenConcat::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(nbOutputs == 1);
    mCHW = inputDims[0];
    mNumInputs = nbInputs;
    ASSERT(inputDims[0].nbDims == 3);

    mInputConcatAxis.resize(mNumInputs);
    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputDims[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputDims[i].d[0] == inputDims[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputDims[i].d[1] == inputDims[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputDims[i].d[2] == inputDims[0].d[2]);
        }
        flattenInput = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    mCopySize.resize(mNumInputs);
    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2] * sizeof(float);
    }
}

void FlattenConcatDynamic::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    ASSERT(nbOutputs == 1);
    
    mNCHW = in[0].desc.dims;
    mNumInputs = nbInputs;
    
    // Verify all the output dimensions
    for (int i = 0; i < nbOutputs; i++)
    {
        ASSERT(out[i].desc.dims.nbDims == 4);
    }

    mInputConcatAxis.resize(mNumInputs);
    // Verify all the input dimensions
    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        ASSERT(in[i].desc.dims.nbDims == 4);
        if (mConcatAxisID != 1)
        {
            ASSERT(in[i].desc.dims.d[1] == in[0].desc.dims.d[1]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(in[i].desc.dims.d[2] == in[0].desc.dims.d[2]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(in[i].desc.dims.d[3] == in[0].desc.dims.d[3]);
        }
        flattenInput = in[i].desc.dims.d[1] * in[i].desc.dims.d[2] * in[i].desc.dims.d[3];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    mCopySize.resize(mNumInputs);
    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = in[i].desc.dims.d[1] * in[i].desc.dims.d[2] * in[i].desc.dims.d[3] * sizeof(float);
    }
}

bool FlattenConcat::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
bool FlattenConcatDynamic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // mNumInputs inputs, 1 outputs, so mNumInputs+1 input/output in total
    ASSERT(0 <= pos && pos < mNumInputs+1);
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR);
}

const char* FlattenConcat::getPluginType() const
{
    return FLATTENCONCAT_PLUGIN_NAMES[0];
}
const char* FlattenConcatDynamic::getPluginType() const noexcept
{
    return FLATTENCONCAT_PLUGIN_NAMES[1];
}

const char* FlattenConcat::getPluginVersion() const
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

const char* FlattenConcatDynamic::getPluginVersion() const noexcept
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

void FlattenConcat::destroy()
{
    delete this;
}

void FlattenConcatDynamic::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* FlattenConcat::clone() const
{
    auto* plugin = new FlattenConcat(
        mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis.data(), mCopySize.data());
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* FlattenConcatDynamic::clone() const noexcept
{
    auto* plugin = new FlattenConcatDynamic(
        mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis.data(), mCopySize.data());
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

FlattenConcatPluginCreator::FlattenConcatPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FlattenConcatPluginCreator::getPluginName() const
{
    return FLATTENCONCAT_PLUGIN_NAMES[0];
}

const char* FlattenConcatPluginCreator::getPluginVersion() const
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

const PluginFieldCollection* FlattenConcatPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* FlattenConcatPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "axis"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mConcatAxisID = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "ignoreBatch"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
        }
    }

    auto* plugin = new FlattenConcat(mConcatAxisID, mIgnoreBatch);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* FlattenConcatPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2Ext* plugin = new FlattenConcat(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}


FlattenConcatDynamicPluginCreator::FlattenConcatDynamicPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FlattenConcatDynamicPluginCreator::getPluginName() const
{
    return FLATTENCONCAT_PLUGIN_NAMES[1];
}

const char* FlattenConcatDynamicPluginCreator::getPluginVersion() const
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

const PluginFieldCollection* FlattenConcatDynamicPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* FlattenConcatDynamicPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "axis"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mConcatAxisID = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "ignoreBatch"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
        }
    }

    auto* plugin = new FlattenConcatDynamic(mConcatAxisID, mIgnoreBatch);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* FlattenConcatDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2DynamicExt* plugin = new FlattenConcatDynamic(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
