#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageProcessor {
public:
    static bool preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW);
    static void createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
};