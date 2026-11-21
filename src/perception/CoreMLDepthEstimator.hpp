#pragma once
#include "perception/DepthModel.hpp"
#include <memory>
#include <string>

// MiDaS through Core ML, allowed on the CPU and Neural Engine. Uses a model built in
// coreml/ (default: the 8-bit-weight W8 version), which runs fully on the Neural Engine.
//
// The model takes a 256x256 BGR image and normalizes it itself, and returns a 256x256
// Float16 relative inverse depth image, so this class only resizes, copies and scales.
class CoreMLDepthEstimator : public DepthModel {
public:
    explicit CoreMLDepthEstimator(const std::string& modelPath);
    ~CoreMLDepthEstimator() override;

    bool load() override;
    bool isLoaded() const override;
    cv::Mat infer(const cv::Mat& bgr) override;
    std::string name() const override { return "Core ML Neural Engine"; }

private:
    struct Impl;  // holds the Objective-C objects, defined in the .mm file
    std::unique_ptr<Impl> impl_;
};
