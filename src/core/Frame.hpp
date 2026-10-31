#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>

struct Frame {
    cv::Mat   image;
    cv::Mat   depth;         // float32 single-channel [0,1], 1 = closest; raw MiDaS output
    cv::Mat   depthSmoothed; // float32 single-channel [0,1]; bilateral + EMA filtered
    cv::Mat   composited;    // BGRA CV_8UC4; camera + occluded sphere, owned by the renderer
    double    timestamp = 0.0; // seconds, steady_clock epoch
    uint64_t  id        = 0;
};
