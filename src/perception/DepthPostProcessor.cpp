#include "perception/DepthPostProcessor.hpp"
#include <opencv2/imgproc.hpp>

DepthPostProcessor::DepthPostProcessor(float  alpha,
                                       int    bilateralD,
                                       double sigmaColor,
                                       double sigmaSpace,
                                       float  motionScale)
    : alpha_(alpha)
    , bilateralD_(bilateralD)
    , sigmaColor_(sigmaColor)
    , sigmaSpace_(sigmaSpace)
    , motionScale_(motionScale)
{}

cv::Mat DepthPostProcessor::process(const cv::Mat& depth) {
    // Bilateral filter requires a contiguous CV_32F mat
    cv::Mat filtered;
    cv::bilateralFilter(depth, filtered, bilateralD_, sigmaColor_, sigmaSpace_);

    if (prev_.empty()) {
        prev_ = filtered.clone();
        return prev_;
    }

    // Per-pixel adaptive alpha: pixels that changed a lot get alpha closer to 1
    // so they respond quickly instead of ghosting.
    // adaptiveAlpha = clamp(alpha_ + motionScale_ * |filtered - prev_|, alpha_, 1)
    cv::Mat delta;
    cv::absdiff(filtered, prev_, delta);
    cv::Mat adaptiveAlpha;
    cv::scaleAdd(delta, static_cast<double>(motionScale_),
                 cv::Mat(delta.size(), CV_32F, cv::Scalar(alpha_)),
                 adaptiveAlpha);
    cv::min(adaptiveAlpha, 1.0f, adaptiveAlpha);

    // EMA: smoothed = prev_ + adaptiveAlpha * (filtered - prev_)
    cv::Mat diff, smoothed;
    cv::subtract(filtered, prev_, diff);
    cv::multiply(adaptiveAlpha, diff, diff);
    cv::add(prev_, diff, smoothed);

    prev_ = smoothed;
    return smoothed;
}

void DepthPostProcessor::reset() {
    prev_.release();
}
