#pragma once

#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvOnnxParser.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    bool buildEngine(const std::string& onnxFile, const std::string& engineFileName);
    bool runInference(const std::string& imageFile);

private:
    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<int64_t> bufferSizes;
    int inputH, inputW, numClasses;

    void readFile(const std::string& fileName, std::vector<char>& buffer);
    void cleanup();
};