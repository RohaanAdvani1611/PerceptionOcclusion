#pragma once
#include <opencv2/opencv.hpp>

// Prepares the real depth map for the per-pixel occlusion test in the Metal
// fragment shader, which also does the compositing.
class OcclusionPass {
public:
    // realDepth : CV_32F [0,1] — MiDaS smoothed depth, 1=closest to camera
    // out       : CV_32F target, written in place (the renderer's GPU-visible buffer)
    void apply(const cv::Mat& realDepth, cv::Mat out) const;
};
