#pragma once
#include <opencv2/opencv.hpp>

struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

class Camera {
public:
    explicit Camera(int index = 0);
    bool open();
    bool read(cv::Mat& frame);
    void release();

    void setResolution(int width, int height);
    void setFPS(int fps);
    double getActualFPS() const;
    CameraIntrinsics getIntrinsics() const;

private:
    int index_;
    int width_  = 0;
    int height_ = 0;
    cv::VideoCapture cap_;
};
