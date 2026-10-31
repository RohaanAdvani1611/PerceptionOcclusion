#pragma once
#include <opencv2/opencv.hpp>

class DepthPostProcessor {
public:
    // alpha       — base EMA weight for current frame (0=fully smooth, 1=no smoothing)
    // bilateralD  — bilateral filter pixel diameter
    // sigmaColor  — depth-difference tolerance for bilateral, in depth units [0, 1]
    //               (larger = blur across deeper edges)
    // sigmaSpace  — spatial falloff for bilateral, in pixels
    // motionScale — how aggressively per-pixel alpha rises with depth change magnitude
    //               (alpha_effective = clamp(alpha + motionScale * |delta|, alpha, 1))
    explicit DepthPostProcessor(float  alpha       = 0.3f,
                                int    bilateralD  = 9,
                                double sigmaColor  = 0.1,
                                double sigmaSpace  = 3.0,
                                float  motionScale = 5.0f);

    // Applies bilateral spatial filter then EMA temporal smoothing.
    // Input/output: float32 single-channel [0, 1].
    cv::Mat process(const cv::Mat& depth);

    void reset(); // discard temporal history (e.g. on scene cut)

private:
    float  alpha_;
    int    bilateralD_;
    double sigmaColor_;
    double sigmaSpace_;
    float  motionScale_;

    cv::Mat prev_; // last smoothed frame
};
