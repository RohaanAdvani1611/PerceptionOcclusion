#pragma once
#include <opencv2/core.hpp>
#include <string>

// What the engine needs from a depth model. DepthEstimator runs the ONNX model on the CPU,
// CoreMLDepthEstimator runs the Core ML model on the Neural Engine.
class DepthModel {
public:
    virtual ~DepthModel() = default;

    virtual bool load() = 0;
    virtual bool isLoaded() const = 0;

    // Returns float32 single-channel depth map, same spatial size as input bgr.
    // Values normalized to [0, 1] where 1 = closest to camera.
    virtual cv::Mat infer(const cv::Mat& bgr) = 0;

    // Short label for logs and the overlay, e.g. "ONNX CPU"
    virtual std::string name() const = 0;
};

// Upsamples raw model output to outSize and maps its 1st..99th percentile range to [0, 1].
// Percentile clipping is robust against outlier pixels that would otherwise collapse the
// whole depth scale when they briefly dominate min/max. Shared so both models match.
cv::Mat normalizeDepth(const cv::Mat& raw, cv::Size outSize);
