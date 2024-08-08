#include "ImageProcessor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

bool ImageProcessor::preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW) {
    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Failed to read image " << imageFile << std::endl;
        return false;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(buffer + i * inputH * inputW, channels[i].data, inputH * inputW * sizeof(float));
    }
    return true;
}



void softmax(float* input, int size) {
    float max = *std::max_element(input, input + size);
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = std::exp(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses) {
    int outputH = inputH;
    int outputW = inputW;

    cv::Mat segmentationMask(outputH, outputW, CV_8UC1);

    std::vector<int> classCounts(numClasses, 0);

    for (int y = 0; y < outputH; ++y) {
        for (int x = 0; x < outputW; ++x) {
            std::vector<float> pixelProbs(numClasses);
            for (int c = 0; c < numClasses; ++c) {
                pixelProbs[c] = outputBuffer[(c * outputH * outputW) + (y * outputW) + x];
            }
            
            // Apply softmax
            softmax(pixelProbs.data(), numClasses);

            int maxClass = std::max_element(pixelProbs.begin(), pixelProbs.end()) - pixelProbs.begin();
            segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
            
            classCounts[maxClass]++;
        }
    }

    cv::Mat resizedMask;
    cv::resize(segmentationMask, resizedMask, cv::Size(inputW, inputH), 0, 0, cv::INTER_NEAREST);

    cv::Mat colorMask;
    cv::applyColorMap(resizedMask, colorMask, cv::COLORMAP_JET);

    cv::imwrite("segmentation_mask.png", colorMask);
    std::cout << "Segmentation mask saved as 'segmentation_mask.png'" << std::endl;

    cv::imwrite("segmentation_mask_gray.png", resizedMask);
    std::cout << "Grayscale segmentation mask saved as 'segmentation_mask_gray.png'" << std::endl;
}