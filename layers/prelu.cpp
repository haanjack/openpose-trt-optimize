#include "prelu.h"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "common.h"

#include "kernel_func.cuh"
#include <cstdio>

using namespace nvinfer1;
using namespace plugin;

PReLUPlugin::PReLUPlugin(const Weights *weights, int nbWeights)
{
    assert(nbWeights == 1);
    mWeights = weights[0];
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), weights[0].values, mWeights.count * type2size(mWeights.type));
}

// create the plugin at runtime from a byte stream
PReLUPlugin::PReLUPlugin(const void *data, size_t length)
{
    const char *d = static_cast<const char *>(data), *a = d;
    read<int>(d, mNbInputChannels);
    read<int>(d, mNbInputHeight);
    read<int>(d, mNbInputWidth);
    read<int>(d, mNbInputCount);
    read<bool>(d, mChannelShared);
    read<int64_t>(d, mWeights.count);
    read<DataType>(d, mWeights.type);

    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type)); //deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void *>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    deserializeToDevice(d, mDeviceKernel, mWeights.count * type2size(mWeights.type));

    assert(d == a + length);
}

PReLUPlugin::~PReLUPlugin()
{
    if (mWeights.values) 
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
    if (mDeviceKernel) 
    {
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}

int PReLUPlugin::getNbOutputs() const
{
    return 1;
}

Dims PReLUPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

bool PReLUPlugin::supportsFormat(DataType type, PluginFormat format) const { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }

void PReLUPlugin::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
    assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);

    mNbInputChannels = inputDims[0].d[0]; 
    mNbInputHeight = inputDims[0].d[1];
    mNbInputWidth = inputDims[0].d[2];
    mNbInputCount = mNbInputChannels * mNbInputHeight * mNbInputWidth;
    mWeights.type = type;
}

int PReLUPlugin::initialize()
{
    cudaMalloc(&mDeviceKernel, mWeights.count * type2size(mWeights.type));
    cudaMemcpy(mDeviceKernel, mWeights.values, mWeights.count * type2size(mWeights.type), cudaMemcpyHostToDevice);
    return 0;
}

void PReLUPlugin::terminate()
{
    if (mWeights.values)
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
    if (mDeviceKernel)
    {
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}

size_t PReLUPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int PReLUPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    const float onef{1.0f}, zerof{0.0f};
    const __half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

    const int count = batchSize * mNbInputCount;
    const int channels = mNbInputChannels;
    const int dim = mNbInputHeight * mNbInputHeight;
    const int div_factor = mChannelShared ? mNbInputChannels : 1; // mChannelShared default is false

    if (mWeights.type == DataType::kFLOAT)
    {
        CHECK(Forward_gpu<float>(count, channels, dim,
                                 reinterpret_cast<const float *>(mDeviceKernel),
                                 reinterpret_cast<const float *>(inputs[0]),
                                 reinterpret_cast<float *>(outputs[0]),
                                 zerof,
                                 div_factor));
    }
    else
    {
        CHECK(Forward_gpu<__half>(count, channels, dim,
                                  reinterpret_cast<const __half *>(mDeviceKernel),
                                  reinterpret_cast<const __half *>(inputs[0]),
                                  reinterpret_cast<__half *>(outputs[0]),
                                  zeroh,
                                  div_factor));
    }

    return 0;
}

size_t PReLUPlugin::getSerializationSize()
{
    return 4 * sizeof(int) + sizeof(bool)
            + sizeof(mWeights.count) + sizeof(mWeights.type) 
            + mWeights.count * type2size(mWeights.type);
}

void PReLUPlugin::serialize(void *buffer)
{
    char *d = static_cast<char *>(buffer), *a = d;

    write(d, mNbInputChannels);
    write(d, mNbInputHeight);
    write(d, mNbInputWidth);
    write(d, mNbInputCount);
    write(d, mChannelShared);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d, mWeights);
    assert(d == a + getSerializationSize());
}

const char *PReLUPlugin::getPluginType() const
{
    return "PReLUPlugin_TRT";
}

const char *PReLUPlugin::getPluginVersion() const
{
    return "001";
}

void PReLUPlugin::destroy() { delete this; }

IPluginExt *PReLUPlugin::clone() const
{
    return new PReLUPlugin(&mWeights, 1);
}

size_t PReLUPlugin::type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

template <typename T>
void PReLUPlugin::write(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void PReLUPlugin::read(const char *&buffer, T &val)
{
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

void *PReLUPlugin::copyToDevice(const void *data, size_t count)
{
    void *deviceData;
    CHECK(cudaMalloc(&deviceData, count));
    CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
    return deviceData;
}

void PReLUPlugin::convertAndCopyToDevice(void *&deviceWeights, const Weights &weights)
{
    if (weights.type != mWeights.type) // Weights are converted in host memory first, if the type does not match
    {
        size_t size = weights.count * (mWeights.type == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
        void *buffer = malloc(size);
        for (int64_t v = 0; v < weights.count; ++v)
            if (mWeights.type == DataType::kFLOAT)
                static_cast<float *>(buffer)[v] = fp16::__half2float(static_cast<const __half *>(weights.values)[v]);
            else
                static_cast<__half *>(buffer)[v] = fp16::__float2half(static_cast<const float *>(weights.values)[v]);

        deviceWeights = copyToDevice(buffer, size);
        free(buffer);
    }
    else
        deviceWeights = copyToDevice(weights.values, weights.count * type2size(mWeights.type));
}

void PReLUPlugin::convertAndCopyToBuffer(char *&buffer, const Weights &weights)
{
    if (weights.type != mWeights.type)
        for (int64_t v = 0; v < weights.count; ++v)
            if (mWeights.type == DataType::kFLOAT)
                reinterpret_cast<float *>(buffer)[v] = fp16::__half2float(static_cast<const __half *>(weights.values)[v]);
            else
                reinterpret_cast<__half *>(buffer)[v] = fp16::__float2half(static_cast<const float *>(weights.values)[v]);
    else
        memcpy(buffer, weights.values, weights.count * type2size(mWeights.type));
    buffer += weights.count * type2size(mWeights.type);
}

void PReLUPlugin::deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size)
{
    deviceWeights = copyToDevice(hostBuffer, size);
    hostBuffer += size;
}

/* Class Plugin Factory Definitions */
// caffe parser plugin implementation
bool PluginFactory::isPlugin(const char *name) 
{
    return isPluginExt(name);
}

bool PluginFactory::isPluginExt(const char *name) 
{
    std::string strName{name};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);
    return (strName.find("prelu") != std::string::npos);
}

IPlugin *PluginFactory::createPlugin(const char *layerName, const Weights* weights, int nbWeights) 
{
    assert(isPlugin(layerName));

    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("prelu") != std::string::npos)
    {
        _nvPlugins[layerName] = (IPlugin *)(new PReLUPlugin(weights, nbWeights));
        return _nvPlugins.at(layerName);
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

// deserialization plugin implementation
IPlugin *PluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    assert(isPlugin(layerName));

    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("prelu") != std::string::npos)
    {
        _nvPlugins[layerName] = (IPlugin *)(new PReLUPlugin(serialData, serialLength));
        return _nvPlugins.at(layerName);
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

// User application destroys plugin when it is safe to do so.
// Should be done after consumers of plugin (like ICudaEngine) are destroyed.
void PluginFactory::destroyPlugin()
{
    try
    {
        for (auto it = _nvPlugins.begin(); it != _nvPlugins.end(); it++)
        {
            if (strstr(it->first.c_str(), "prelu"))
            {
                delete (PReLUPlugin *)(it->second);
            }
            _nvPlugins.erase(it);
        }
    }
    catch (...)
    {
        std::exception_ptr p = std::current_exception();
        std::clog << (p ? p.__cxa_exception_type()->name() : "null") << std::endl;
    }
}

