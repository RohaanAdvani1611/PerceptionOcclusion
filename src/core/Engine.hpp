#pragma once
#include "core/Frame.hpp"
#include "core/Profiler.hpp"
#include "camera/Camera.hpp"
#include "perception/DepthEstimator.hpp"
#include "perception/DepthPostProcessor.hpp"
#include "rendering/MetalContext.hpp"
#include "rendering/Renderer.hpp"
#include "compositing/OcclusionPass.hpp"
#include <string>

struct EngineOptions {
    int         benchFrames = 0;  // >0: exit after this many measured frames (post warm-up)
    std::string csvPath;          // non-empty: write per-frame timings here on exit
};

class Engine {
public:
    explicit Engine(EngineOptions opts = {});
    void run();

private:
    bool initSystems();
    bool processFrame(Frame& frame);
    void updateDisplay(Frame& frame);
    void handleInput(int key);

    EngineOptions      opts_;
    Profiler           profiler_;
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
