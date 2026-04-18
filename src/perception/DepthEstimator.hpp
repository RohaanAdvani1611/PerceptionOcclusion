#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

class DepthEstimator {
public:
    explicit DepthEstimator(const std::string& modelPath);

    bool load();
    bool isLoaded() const { return loaded_; }

    // Returns float32 single-channel depth map, same spatial size as input bgr.
    // Values normalized to [0, 1] where 1 = closest to camera.
    cv::Mat infer(const cv::Mat& bgr);

    int inputWidth()  const { return inputW_; }
    int inputHeight() const { return inputH_; }

private:
    std::string modelPath_;
    bool        loaded_ = false;

    Ort::Env                      env_;
    Ort::SessionOptions           sessionOpts_;
    std::unique_ptr<Ort::Session> session_;

    std::string inputName_;
    std::string outputName_;
    int inputH_ = 384;
    int inputW_ = 384;

    std::vector<float> preprocess(const cv::Mat& bgr) const;
};
