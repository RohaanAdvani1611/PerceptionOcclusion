#include "perception/DepthEstimator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

DepthEstimator::DepthEstimator(const std::string& modelPath)
    : modelPath_(modelPath)
    , env_(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator")
{}

bool DepthEstimator::load() {
    try {
        sessionOpts_.SetIntraOpNumThreads(4);
        sessionOpts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, modelPath_.c_str(), sessionOpts_);

        Ort::AllocatorWithDefaultOptions allocator;
        inputName_  = session_->GetInputNameAllocated(0, allocator).get();
        outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

        // Read spatial dimensions from model metadata if fixed
        auto shape = session_->GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape(); // expected [1, 3, H, W]
        if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
            inputH_ = static_cast<int>(shape[2]);
            inputW_ = static_cast<int>(shape[3]);
        }

        loaded_ = true;
        std::cout << "DepthEstimator: model loaded  input=" << inputW_ << "x" << inputH_ << "\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "DepthEstimator: " << e.what() << "\n";
        return false;
    }
    return true;
}

std::vector<float> DepthEstimator::preprocess(const cv::Mat& bgr) const {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputW_, inputH_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW. Values stay in [0, 1]: the ONNX graph applies ImageNet mean/std
    // itself in its first nodes, so normalizing here would apply them twice.
    std::vector<float> blob(3 * inputH_ * inputW_);
    const int plane = inputH_ * inputW_;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < inputH_; ++y) {
            const float* row = rgb.ptr<float>(y);
            for (int x = 0; x < inputW_; ++x) {
                blob[c * plane + y * inputW_ + x] = row[x * 3 + c];
            }
        }
    }
    return blob;
}

cv::Mat DepthEstimator::infer(const cv::Mat& bgr) {
    if (!loaded_) return {};

    auto blob = preprocess(bgr);

    std::array<int64_t, 4> inputShape{1, 3, inputH_, inputW_};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), inputShape.data(), inputShape.size());

    const char* inNames[]  = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames, 1);

    auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // Output is [1,H,W] or [H,W] or [1,1,H,W] — H and W are always the last two dims
    int outH = static_cast<int>(outShape[outShape.size() - 2]);
    int outW = static_cast<int>(outShape[outShape.size() - 1]);

    const float* outData = outputs[0].GetTensorData<float>();
    cv::Mat depth(outH, outW, CV_32F, const_cast<float*>(outData));
    depth = depth.clone(); // detach from tensor memory before outputs is destroyed

    cv::resize(depth, depth, cv::Size(bgr.cols, bgr.rows), 0, 0, cv::INTER_LINEAR);

    // Percentile-clip normalization: robust against outlier pixels that would
    // otherwise collapse the whole depth scale when they briefly dominate min/max.
    std::vector<float> vals(depth.begin<float>(), depth.end<float>());
    int n     = static_cast<int>(vals.size());
    int loIdx = std::max(0, static_cast<int>(0.01f * n));
    int hiIdx = std::min(n - 1, static_cast<int>(0.99f * n));
    std::nth_element(vals.begin(), vals.begin() + loIdx, vals.end());
    float pLow = vals[loIdx];
    std::nth_element(vals.begin() + loIdx + 1, vals.begin() + hiIdx, vals.end());
    float pHigh = vals[hiIdx];

    if (pHigh > pLow) {
        float range = pHigh - pLow;
        depth.forEach<float>([pLow, range](float& v, const int*) {
            v = std::max(0.0f, std::min(1.0f, (v - pLow) / range));
        });
    }

    return depth;
}
