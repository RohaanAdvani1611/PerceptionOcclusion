#pragma once
#include "perception/DepthModel.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

// MiDaS through ONNX Runtime on the CPU (4 threads)
class DepthEstimator : public DepthModel {
public:
    explicit DepthEstimator(const std::string& modelPath);

    bool load() override;
    bool isLoaded() const override { return loaded_; }
    cv::Mat infer(const cv::Mat& bgr) override;
    std::string name() const override { return "ONNX CPU"; }

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
