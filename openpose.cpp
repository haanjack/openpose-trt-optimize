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


struct Params
{
    std::string deployFile, modelFile, engine, calibrationCache{"CalibrationTable"};
    std::string inputs;
    std::vector<std::string> outputs;
    int device{0}, batchSize{1}, workspaceSize{16}, iterations{10}, avgRuns{10}, useDLA{0};
    bool fp16{false}, int8{false}, verbose{false}, allowGPUFallback{false};
    float pct{99};
} gParams;
std::vector<std::string> gInputs;
std::map<std::string, Dims3> gInputDimensions;

std::string locateFile(const std::string& input)
{
    // std::vector<std::string> dirs{"models/pose/coco/"};
    std::vector<std::string> dirs{"./"};
    return locateFile(input, dirs);
}

/* Logger */
float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
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

ICudaEngine* caffeToTRTModel()
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    
    // Parse Plugin Layers
    PluginFactory parserPluginFactory;
    parser->setPluginFactoryExt(&parserPluginFactory);

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(gParams.deployFile.c_str(),    // caffe deploy file
                      gParams.modelFile.c_str(),     // caffe model file
                      *network,                          // network definition that the parser will populate
                      gParams.fp16 ? DataType::kHALF : DataType::kFLOAT);
    if (!blobNameToTensor)
        return nullptr;

    // TODO: Input??
    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    // the caffe file has no notion of outputs, 
    // so we need to manually say which tensors the engine should generate
    for (auto& s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                  << dims.d[2] << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(size_t(gParams.workspaceSize) << 20);
    builder->setFp16Mode(gParams.fp16);

    // RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    // if (gParams.int8)
    // {
    //     builder->setInt8Mode(true);
    //     builder->setInt8Calibrator(&calibrator);
    // }

    if (gParams.useDLA > 0)
    {
        builder->setDefaultDeviceType(static_cast<DeviceType>(gParams.useDLA));
        if (gParams.allowGPUFallback)
            builder->allowGPUFallback(gParams.allowGPUFallback);
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    // we don't need the network any more, and we can destroy the parser
    parserPluginFactory.destroyPlugin();
    parser->destroy();
    network->destroy();
    builder->destroy();
    
    return engine;
}

void timeInference(ICudaEngine* engine)
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
    size_t inputSize = gParams.batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float);
    size_t outputSize = gParams.batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    // zero the input buffer
    CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

    for (int i = 0; i < TIMING_ITERATIONS;i++)
        context->execute(gParams.batchSize, buffers);

    // release the context and buffers
    context->destroy();
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void createMemory(const ICudaEngine* engine, std::vector<void*>& buffers, const std::string& name)
{
    size_t bindingIndex = engine->getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int) bindingIndex, (int) buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3&&>(engine->getBindingDimensions((int) bindingIndex));
    size_t eltCount = dimensions.d[0] * dimensions.d[1] * dimensions.d[2] * gParams.batchSize, memSize = eltCount * sizeof(float);

    float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
}

void doInference(ICudaEngine* engine)
{
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.

    std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
    for (size_t i = 0; i < gInputs.size(); i++)
        createMemory(engine, buffers, gInputs[i]);

    for (size_t i = 0; i < gParams.outputs.size(); i++)
        createMemory(engine, buffers, gParams.outputs[i]);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float totalGpu{0}, totalHost{0}; // GPU and Host timers
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }
        totalGpu /= gParams.avgRuns;
        totalHost /= gParams.avgRuns;
        std::cout << "Average over " << gParams.avgRuns << " runs is " << totalGpu << " ms (host walltime is " << totalHost
                  << " ms, " << static_cast<int>(gParams.pct) << "\% percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
}

ICudaEngine* createEngine()
{
    ICudaEngine* engine;
    if ((!gParams.deployFile.empty()))
    {
        // Create engine (caffe)
        engine = caffeToTRTModel(); // load prototxt & caffemodel files
        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        // write plan file if it is specified
        if (!gParams.engine.empty())
        {
            std::ofstream p(gParams.engine);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return nullptr;
            }
            IHostMemory* ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
        }
        return engine;
    }

    // load directlry from serialized engine file if deploy not specified
    if (!gParams.engine.empty()) {
        char* trtModelStream {nullptr};
        size_t size{0};
        std::ifstream file(gParams.engine, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        IRuntime* infer = createInferRuntime(gLogger);
        PluginFactory pluginFactory;
        engine = infer->deserializeCudaEngine(trtModelStream, size, &pluginFactory);
        pluginFactory.destroyPlugin();
        if (trtModelStream) delete[] trtModelStream;

        gParams.inputs.empty() ?
            gInputs.push_back("image") :
            gInputs.push_back(gParams.inputs.c_str());
        return engine;
    }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
}

static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>      Caffe deploy file\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");
    printf("  --model=<file>       Caffe model file (default = no model, random weights used)\n");

    printf("\nOptional params:\n");

    printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --percentile=P       For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 representing min, and 100 representing max; default = %.1f%%)\n", gParams.pct);
    printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --fp16               Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8               Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose            Use verbose logging (default = false)\n");
    printf("  --engine=<file>      Generate a serialized TensorRT engine\n");
    printf("  --calib=<file>       Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf("  --useDLA=N           Enable execution on DLA for all layers that support dla. Value can range from 1 to N, where N is the number of dla engines on the platform.\n");
    printf("  --allowGPUFallback   If --useDLA flag is present and if a layer can't run on DLA, then run on GPU. \n");
    fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        std::cout << name << std::endl;
        value = true;
    }
    return match;
}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile) || parseString(argv[j], "engine", gParams.engine))
            continue;

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns)
            || parseInt(argv[j], "device", gParams.device) || parseInt(argv[j], "workspace", gParams.workspaceSize)
            || parseInt(argv[j], "useDLA", gParams.useDLA))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        if (parseBool(argv[j], "fp16", gParams.fp16) || parseBool(argv[j], "int8", gParams.int8)
            || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "allowGPUFallback", gParams.allowGPUFallback))
            continue;

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (!parseArgs(argc, argv))
        return -1;

    cudaSetDevice(gParams.device);

    if (gParams.outputs.size() == 0 && !gParams.deployFile.empty() && !gParams.modelFile.empty())
    {
        std::cerr << "At least one network output must be defined" << std::endl;
        return -1;
    }

    std::cout << "Building and running a GPU inference engine for OpenPose, N=" << gParams.batchSize << "..." << std::endl;
    
    // create an engine
    ICudaEngine* engine = createEngine();
    if (!engine)
    {
        std::cerr << "Engine could not be created" << std::endl;
        return -1;  
    }
    nvcaffeparser1::shutdownProtobufLibrary();

    // run inference with null data to time network performance
    std::cout << "Run inference..." << std::endl;
    // timeInference(engine);
    doInference(engine);

    engine->destroy();

    // gProfiler.printLayerTimes();

    std::cout << "Done." << std::endl;

    return 0;
}
