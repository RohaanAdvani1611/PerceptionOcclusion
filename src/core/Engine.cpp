#include "core/Engine.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <csignal>

static std::atomic<bool> gQuit{false};
static void onSignal(int) { gQuit = true; }

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ── Construction ──────────────────────────────────────────────────────────────

Engine::Engine()
    : camera_(1), depthEstimator_(PROJECT_ROOT "/assets/models/midas.onnx"), renderer_(ctx_)
{
}

// ── Private helpers ───────────────────────────────────────────────────────────

bool Engine::initSystems()
{
    std::cout << std::fixed << std::setprecision(1);

    if (!camera_.open())
        return false;
    camera_.setResolution(640, 480);

    hasDepth_ = depthEstimator_.load();
    if (!hasDepth_)
        std::cerr << "Engine: depth disabled — place midas.onnx in assets/models/\n";

    hasMetal_ = ctx_.init(PROJECT_ROOT "/assets/shaders") &&
                renderer_.init(640, 480, camera_.getIntrinsics());
    if (!hasMetal_)
        std::cerr << "Engine: rendering disabled — Metal init failed\n";

    std::cout << "Camera hardware FPS: " << camera_.getActualFPS() << "\n";
    std::cout << "Controls  move sphere:      U/D=up/down  L/R=left/right  F/B=forward/back\n";
    std::cout << "          occlusion threshold: [ = decrease   ] = increase\n";
    std::cout << "          quit: ESC or Q\n";
    return true;
}

bool Engine::processFrame(Frame &frame, double &readMs, double &inferMs, double &renderMs)
{
    auto t0 = Clock::now();
    if (!camera_.read(frame.image))
        return false;
    auto t1 = Clock::now();

    frame.timestamp = std::chrono::duration<double>(t1.time_since_epoch()).count();
    readMs = Ms(t1 - t0).count();

    if (hasDepth_)
    {
        auto ti = Clock::now();
        frame.depth = depthEstimator_.infer(frame.image);
        frame.depthSmoothed = depthPostProcessor_.process(frame.depth);
        inferMs = Ms(Clock::now() - ti).count();
    }

    if (hasMetal_)
    {
        auto tr = Clock::now();
        frame.rendered = renderer_.render();
        renderMs = Ms(Clock::now() - tr).count();
    }

    if (hasMetal_ && hasDepth_)
        frame.occluded = occlusionPass_.apply(
            frame.rendered, frame.depthSmoothed, occlusionThreshold_);

    return true;
}

void Engine::updateDisplay(const Frame &frame, double fps, double &showMs)
{
    auto t0 = Clock::now();

    cv::Mat camView = frame.image.clone();
    cv::putText(camView, "FPS: " + std::to_string(static_cast<int>(fps)),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 255, 0), 2);
    // cv::imshow("Camera", camView);

    if (hasDepth_ && !frame.depthSmoothed.empty())
    {
        cv::Mat depthVis;
        frame.depthSmoothed.convertTo(depthVis, CV_8U, 255.0);
        cv::applyColorMap(depthVis, depthVis, cv::COLORMAP_INFERNO);
        cv::imshow("Depth Map", depthVis);
    }

    // Use occluded result when available, fall back to raw render
    const cv::Mat &composite = (!frame.occluded.empty()) ? frame.occluded : frame.rendered;
    if (hasMetal_ && !composite.empty())
    {
        std::vector<cv::Mat> ch(4);
        cv::split(composite, ch); // [B, G, R, A]
        cv::Mat sphereBGR;
        cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, sphereBGR);
        cv::Mat arView = frame.image.clone();
        sphereBGR.copyTo(arView, ch[3]);
        cv::putText(arView,
                    "Threshold: " + std::to_string(occlusionThreshold_).substr(0, 4),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2);
        cv::imshow("AR View", arView);
    }

    showMs = Ms(Clock::now() - t0).count();
}

void Engine::handleInput(int key)
{
    if (!hasMetal_)
        return;

    constexpr float kStep = 0.05f;
    bool moved = true;
    switch (key)
    {
    case 'u':
    case 'U':
        renderer_.moveSphere(0, +kStep, 0);
        break;
    case 'd':
    case 'D':
        renderer_.moveSphere(0, -kStep, 0);
        break;
    case 'l':
    case 'L':
        renderer_.moveSphere(-kStep, 0, 0);
        break;
    case 'r':
    case 'R':
        renderer_.moveSphere(+kStep, 0, 0);
        break;
    case 'f':
    case 'F':
        renderer_.moveSphere(0, 0, -kStep);
        break;
    case 'b':
    case 'B':
        renderer_.moveSphere(0, 0, +kStep);
        break;
    default:
        moved = false;
        break;
    }

    if (moved)
    {
        std::cout << "Sphere  x=" << renderer_.sphereX()
                  << "  y=" << renderer_.sphereY()
                  << "  z=" << renderer_.sphereZ() << "\n";
        return;
    }

    constexpr float kTStep = 0.05f;
    if (key == '[')
    {
        occlusionThreshold_ = std::max(0.05f, occlusionThreshold_ - kTStep);
        std::cout << "Occlusion threshold: " << occlusionThreshold_ << "\n";
    }
    else if (key == ']')
    {
        occlusionThreshold_ = std::min(0.95f, occlusionThreshold_ + kTStep);
        std::cout << "Occlusion threshold: " << occlusionThreshold_ << "\n";
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void Engine::run()
{
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    if (!initSystems())
        return;

    Frame frame;
    uint64_t frameId = 0;
    int fpsTick = 0;
    double fps = 0.0;
    double readMs = 0.0;
    double inferMs = 0.0;
    double renderMs = 0.0;
    double showMs = 0.0;
    auto fpsTimer = Clock::now();

    while (!gQuit)
    {
        if (!processFrame(frame, readMs, inferMs, renderMs))
            break;
        frame.id = frameId++;

        ++fpsTick;
        double elapsed = Ms(Clock::now() - fpsTimer).count() / 1000.0;
        if (elapsed >= 1.0)
        {
            fps = fpsTick / elapsed;
            fpsTick = 0;
            fpsTimer = Clock::now();
            std::cout << "FPS: " << fps
                      << "  read: " << readMs << "ms"
                      << "  infer: " << inferMs << "ms"
                      << "  render: " << renderMs << "ms"
                      << "  show: " << showMs << "ms\n";
        }

        updateDisplay(frame, fps, showMs);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        handleInput(key);
    }

    camera_.release();
}
