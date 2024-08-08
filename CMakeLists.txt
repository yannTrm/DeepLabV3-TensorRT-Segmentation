cmake_minimum_required(VERSION 3.10)

project(test_tensorrt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${CUDA_LIBRARIES})
link_directories(${TENSORRT_LIB_DIR})

# Add source files
set(SOURCES
    src/main.cpp
    src/TensorRTEngine.cpp
    src/ImageProcessor.cpp
)

# Add executable
add_executable(main ${SOURCES})

# Include directories for the target
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(main
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)