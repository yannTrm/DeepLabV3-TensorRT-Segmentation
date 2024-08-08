#include "TensorRTEngine.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model> <image> <engine_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string onnxFile = argv[1];
    const std::string imageFile = argv[2];
    const std::string engineFileName = argv[3];

    TensorRTEngine engine;

    if (!engine.buildEngine(onnxFile, engineFileName)) {
        std::cerr << "Failed to build engine" << std::endl;
        return EXIT_FAILURE;
    }

    if (!engine.runInference(imageFile)) {
        std::cerr << "Failed to run inference" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}