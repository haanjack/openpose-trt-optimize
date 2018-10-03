#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

// Custom Layer
#include "layers/prelu.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int BATCH_SIZE = 4;
static const int TIMING_ITERATIONS = 1000;

const char* INPUT_BLOB_NAME = "image";
const char* OUTPUT_BLOB_NAME = "net_output";

// const char *file_prototxt = "pose_deploy_linevec.prototxt";
// const char *file_caffemodel = "pose_iter_440000.caffemodel";
const char *file_prototxt = "pose_deploy.prototxt";
const char *file_caffemodel = "pose_iter_584000.caffemodel";
const char* model_path = "models/pose/coco";

static int gDLA{0};

std::string locateFile(const std::string& input)
{
    // std::vector<std::string> dirs{"models/pose/coco/"};
    std::vector<std::string> dirs{"models/pose/body_25/"};
    return locateFile(input, dirs);
}

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;

void caffeToTRTModel(const std::string& deployFile,             // name for caffe prototxt
                     const std::string& modelFile,              // name for model
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,                 // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactoryExt* pluginFactory, // factory for plugin layers
                     IHostMemory *&trtModelStream)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactoryExt(pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();
    std::cout << locateFile(deployFile).c_str() << std::endl;
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported

    if (useFp16)
        std::cout << "use fp16.." << std::endl;
    else
        std::cout << "use fp32.." << std::endl;

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(locateFile(deployFile).c_str(),               // caffe deploy file
                                 locateFile(modelFile).c_str(),     // caffe model file
                                 *network,                          // network definition that the parser will populate
                                 modelDataType);
    assert(blobNameToTensor != nullptr);

    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    // set up the network for paired-fp16 format if available
    builder->setFp16Mode(useFp16);

    if (gDLA > 0) samplesCommon::enableDLA(builder, gDLA);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void timeInference(ICudaEngine* engine, int batchSize)
{
    // input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // allocate GPU buffers
    Dims3 inputDims = static_cast<Dims3&&>(engine->getBindingDimensions(inputIndex)), outputDims = static_cast<Dims3&&>(engine->getBindingDimensions(outputIndex));
    size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float);
    size_t outputSize = batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    // zero the input buffer
    CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

    for (int i = 0; i < TIMING_ITERATIONS;i++)
        context->execute(batchSize, buffers);

    // release the context and buffers
    context->destroy();
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    std::cout << "Building and running a GPU inference engine for OpenPose, N=" << BATCH_SIZE << "..." << std::endl;
    gDLA = samplesCommon::parseDLA(argc, argv);

    // create a TensorRT model from the caffe model and serialize it to a stream
    // parse the caffe model and the meanPlugin_PReLUile
    PluginFactory parserPluginFactory;
    IHostMemory *trtModelStream{nullptr};
    caffeToTRTModel(file_prototxt,
                    file_caffemodel,
                    std::vector<std::string>{OUTPUT_BLOB_NAME},
                    BATCH_SIZE,
                    &parserPluginFactory,
                    trtModelStream);
    parserPluginFactory.destroyPlugin();
    assert(trtModelStream != nullptr);

    // create an engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    PluginFactory pluginFactory;
    ICudaEngine *engine = runtime->deserializeCudaEngine(
        trtModelStream->data(),
        trtModelStream->size(),
        &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) {
    printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            } else {
    printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
            }
        }

    // run inference with null data to time network performance
    timeInference(engine, BATCH_SIZE);

    engine->destroy();
    runtime->destroy();

    gProfiler.printLayerTimes();

    // Destroy plugins created by factory
    pluginFactory.destroyPlugin();

    std::cout << "Done." << std::endl;

    return 0;
}
