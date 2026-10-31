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

    // Write this frame's inputs into these before render(). Both wrap memory shared
    // with the GPU, so writing to them directly avoids a separate upload.
    cv::Mat cameraInput() const; // BGRA CV_8UC4
    cv::Mat depthInput()  const; // CV_32F [0,1], 1 = closest

    // Draws the camera frame, then the sphere, hiding sphere pixels where the real
    // surface is closer. occlusionThreshold is the real depth value treated as being
    // the sphere's starting distance. The returned BGRA CV_8UC4 mat wraps the GPU
    // output memory and stays valid until the next render().
    cv::Mat render(float occlusionThreshold);

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

    size_t colorBytesPerRow_ = 0;
    size_t depthBytesPerRow_ = 0;

    void* pipelineState_           = nullptr;
    void* backgroundPipelineState_ = nullptr;
    void* depthStencilState_       = nullptr;
    void* backgroundDepthState_    = nullptr;
    void* vertexBuffer_            = nullptr;
    void* indexBuffer_             = nullptr;
    void* outputBuffer_            = nullptr;
    void* cameraBuffer_            = nullptr;
    void* realDepthBuffer_         = nullptr;
    void* outputTexture_           = nullptr;
    void* cameraTexture_           = nullptr;
    void* realDepthTexture_        = nullptr;
    void* metalDepthTexture_       = nullptr;
};
