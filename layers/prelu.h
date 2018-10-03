#ifndef _PLUGIN_PRELU_H_
#define _PLUGIN_PRELU_H_

#include <iostream>
#include <map>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "kernel_func.cuh"

using namespace nvinfer1;
using namespace plugin;

class PReLUPlugin : public IPluginExt
{
public:
    PReLUPlugin(const Weights *weights, int nbWeights);

    // create the plugin at runtime from a byte stream
    PReLUPlugin(const void *data, size_t length);

    ~PReLUPlugin();
    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;
    
    int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    
    virtual size_t getSerializationSize() override;

    virtual void serialize(void* buffer) override;

    const char* getPluginType() const override;
    
    const char* getPluginVersion() const override;
    
    void destroy();
    
    IPluginExt* clone() const override;
    
private:
    size_t type2size(DataType type);

    template <typename T>
    void write(char *&buffer, const T &val);

    template <typename T>
    void read(const char *&buffer, T &val);

    void *copyToDevice(const void *data, size_t count);
    void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights);

    void convertAndCopyToBuffer(char *&buffer, const Weights &weights);

    void deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size);
    
    int mNbInputChannels, mNbInputHeight, mNbInputWidth, mNbInputCount;
    bool mChannelShared;
    Weights mWeights;
    //DataType mDataType{DataType::kFLOAT};

    void* mDeviceKernel{nullptr};
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override;
    
    bool isPluginExt(const char* name) override;
    
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;

    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin();

    std::map<std::string, IPlugin*> _nvPlugins;
};

#endif // _PLUGIN_PRELU_H_