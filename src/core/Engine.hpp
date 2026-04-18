#pragma once
#include "core/Frame.hpp"
#include "camera/Camera.hpp"
#include "perception/DepthEstimator.hpp"
#include "perception/DepthPostProcessor.hpp"
#include "rendering/MetalContext.hpp"
#include "rendering/Renderer.hpp"
#include "compositing/OcclusionPass.hpp"

class Engine {
public:
    Engine();
    void run();

private:
    bool initSystems();
    bool processFrame(Frame& frame, double& readMs, double& inferMs, double& renderMs);
    void updateDisplay(const Frame& frame, double fps, double& showMs);
    void handleInput(int key);

    Camera             camera_;
    DepthEstimator     depthEstimator_;
    DepthPostProcessor depthPostProcessor_;
    MetalContext       ctx_;
    Renderer           renderer_;
    OcclusionPass      occlusionPass_;

    bool  hasDepth_           = false;
    bool  hasMetal_           = false;
    float occlusionThreshold_ = 0.5f;
};
