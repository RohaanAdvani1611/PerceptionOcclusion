#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>

struct Frame {
    cv::Mat   image;
    cv::Mat   depth;         // float32 single-channel [0,1], 1 = closest; raw MiDaS output
    cv::Mat   depthSmoothed; // float32 single-channel [0,1]; bilateral + EMA filtered
    cv::Mat   rendered;      // BGRA CV_8UC4; raw sphere render, alpha=255 where sphere
    cv::Mat   occluded;      // BGRA CV_8UC4; sphere with real-depth occlusion mask applied
    double    timestamp = 0.0; // seconds, steady_clock epoch
    uint64_t  id        = 0;
};
