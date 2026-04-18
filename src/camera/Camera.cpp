#include "camera/Camera.hpp"
#include <iostream>

Camera::Camera(int index) : index_(index) {}

bool Camera::open() {
    cap_.open(index_, cv::CAP_AVFOUNDATION);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera at index " << index_ << "\n";
        return false;
    }
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    width_  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return true;
}

bool Camera::read(cv::Mat& frame) {
    cap_ >> frame;
    return !frame.empty();
}

void Camera::release() {
    cap_.release();
}

void Camera::setResolution(int width, int height) {
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    width_  = width;
    height_ = height;
}

void Camera::setFPS(int fps) {
    cap_.set(cv::CAP_PROP_FPS, fps);
}

double Camera::getActualFPS() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

CameraIntrinsics Camera::getIntrinsics() const {
    // Approximate intrinsics assuming ~60 degree horizontal FOV
    double fx = width_;
    double fy = width_;
    double cx = width_  / 2.0;
    double cy = height_ / 2.0;
    return {fx, fy, cx, cy};
}
