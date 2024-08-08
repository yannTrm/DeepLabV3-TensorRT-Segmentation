#include "TensorRTEngine.hpp"
#include "ImageProcessor.hpp"
#include <fstream>
#include <iostream>
#include "cuda_runtime_api.h"

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

TensorRTEngine::TensorRTEngine() : runtime(nullptr), engine(nullptr), context(nullptr) {}

TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

void TensorRTEngine::readFile(const std::string& fileName, std::vector<char>& buffer) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    
    std::vector<char> modelData;
    readFile(onnxFile, modelData);
    if (!parser->parse(modelData.data(), modelData.size())) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1GB

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "Failed to build serialized engine" << std::endl;
        return false;
    }

    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "Failed to create CUDA engine" << std::endl;
        return false;
    }

    context = engine->createExecutionContext();

    // Get input dimensions
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];

    // Get number of classes
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    numClasses = outputDims.d[1];

    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

bool TensorRTEngine::runInference(const std::string& imageFile) {
    int32_t nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    buffers.resize(nIO);
    bufferSizes.resize(nIO);

    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nvinfer1::Dims dims = context->getTensorShape(vTensorName[i].c_str());
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j)
            size *= dims.d[j];
        bufferSizes[i] = size * sizeof(float);

        cudaMalloc(&buffers[i], bufferSizes[i]);
    }

    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(imageFile, inputBuffer, inputH, inputW)) {
        std::cerr << "Failed to preprocess image" << std::endl;
        return false;
    }
    cudaMemcpy(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice);

    for (int i = 0; i < nIO; ++i) { // defining tensor adress 
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to perform inference" << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);

    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    cudaMemcpy(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);

    ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);

    delete[] inputBuffer;
    delete[] outputBuffer;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);

    return true;
}

void TensorRTEngine::cleanup() {
    if (context) {
        delete context;
        context = nullptr;
    }
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
}