#pragma once
#include "rendering/MetalContext.hpp"
#include "camera/Camera.hpp"
#include <opencv2/opencv.hpp>

// Pure C++ header — Metal types are hidden behind void* in Renderer.mm.
class Renderer {
public:
    explicit Renderer(MetalContext& ctx);
    ~Renderer();

    bool init(int width, int height, const CameraIntrinsics& intrinsics);
    bool isReady() const { return ready_; }

    // Move sphere by delta in camera space (metres)
    void  moveSphere(float dx, float dy, float dz);
    float sphereX() const { return sphereX_; }
    float sphereY() const { return sphereY_; }
    float sphereZ() const { return sphereZ_; }

    // Renders a procedural sphere. Returns BGRA CV_8UC4 mat,
    // alpha=255 where sphere geometry exists, alpha=0 elsewhere.
    cv::Mat render();

    // GPU execution time of the last render() command buffer, excluding readback
    double lastGpuMs() const { return lastGpuMs_; }

private:
    MetalContext&    ctx_;
    bool             ready_      = false;
    int              width_      = 0;
    int              height_     = 0;
    int              indexCount_ = 0;
    CameraIntrinsics intrinsics_ {};
    double           lastGpuMs_  = 0.0;

    float sphereX_ =  0.0f;
    float sphereY_ =  0.0f;
    float sphereZ_ = -1.5f;

    void* pipelineState_     = nullptr;
    void* depthStencilState_ = nullptr;
    void* vertexBuffer_      = nullptr;
    void* indexBuffer_       = nullptr;
    void* colorTexture_      = nullptr;
    void* metalDepthTexture_ = nullptr;
};
