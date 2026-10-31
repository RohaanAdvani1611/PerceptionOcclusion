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

// Frames excluded from the run summary while ONNX Runtime and Metal warm up
constexpr int kWarmupFrames = 30;

// ── Construction ──────────────────────────────────────────────────────────────

Engine::Engine(EngineOptions opts)
    : opts_(std::move(opts)),
      profiler_({"read", "infer", "post", "render", "gpu", "occlusion", "display", "latency"},
                kWarmupFrames),
      camera_(0), depthEstimator_(PROJECT_ROOT "/assets/models/midas.onnx"), renderer_(ctx_)
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

bool Engine::processFrame(Frame &frame)
{
    // read includes blocking on the camera's next frame, so it tracks sensor FPS
    auto t0 = Clock::now();
    if (!camera_.read(frame.image))
        return false;
    auto t1 = Clock::now();

    frame.timestamp = std::chrono::duration<double>(t1.time_since_epoch()).count();
    profiler_.record("read", Ms(t1 - t0).count());

    if (hasDepth_)
    {
        auto ti = Clock::now();
        frame.depth = depthEstimator_.infer(frame.image);
        auto tp = Clock::now();
        frame.depthSmoothed = depthPostProcessor_.process(frame.depth);
        profiler_.record("infer", Ms(tp - ti).count());
        profiler_.record("post", Ms(Clock::now() - tp).count());
    }

    if (hasMetal_ && hasDepth_)
    {
        auto to = Clock::now();
        occlusionPass_.apply(frame.depthSmoothed, renderer_.depthInput());
        profiler_.record("occlusion", Ms(Clock::now() - to).count());
    }

    if (hasMetal_)
    {
        auto tr = Clock::now();
        cv::Mat cameraInput = renderer_.cameraInput();
        if (frame.image.size() == cameraInput.size())
        {
            cv::cvtColor(frame.image, cameraInput, cv::COLOR_BGR2BGRA);
        }
        else
        {
            cv::Mat resized;
            cv::resize(frame.image, resized, cameraInput.size());
            cv::cvtColor(resized, cameraInput, cv::COLOR_BGR2BGRA);
        }
        frame.composited = renderer_.render(occlusionThreshold_);
        profiler_.record("render", Ms(Clock::now() - tr).count());
        profiler_.record("gpu", renderer_.lastGpuMs());
    }

    return true;
}

void Engine::updateDisplay(Frame &frame)
{

    if (hasDepth_ && !frame.depthSmoothed.empty())
    {
        cv::Mat depthVis;
        frame.depthSmoothed.convertTo(depthVis, CV_8U, 255.0);
        cv::applyColorMap(depthVis, depthVis, cv::COLORMAP_INFERNO);
        cv::imshow("Depth Map", depthVis);
    }

    if (hasMetal_ && !frame.composited.empty())
    {
        cv::putText(frame.composited,
                    "Threshold: " + std::to_string(occlusionThreshold_).substr(0, 4),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255, 255), 2);
        cv::imshow("AR View", frame.composited);
    }
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
    auto fpsTimer = Clock::now();

    while (!gQuit)
    {
        if (!processFrame(frame))
            break;
        frame.id = frameId++;

        // waitKey is where highgui actually presents the windows on macOS
        auto td = Clock::now();
        updateDisplay(frame);
        int key = cv::waitKey(1) & 0xFF;
        auto shown = Clock::now();
        profiler_.record("display", Ms(shown - td).count());

        // Processing latency: camera frame handed to us → frame presented
        double shownSec = std::chrono::duration<double>(shown.time_since_epoch()).count();
        profiler_.record("latency", (shownSec - frame.timestamp) * 1000.0);
        profiler_.endFrame();

        double elapsed = Ms(shown - fpsTimer).count() / 1000.0;
        if (elapsed >= 1.0)
        {
            profiler_.printWindow(elapsed);
            fpsTimer = shown;
        }

        if (opts_.benchFrames > 0 && profiler_.measuredFrames() >= opts_.benchFrames)
            break;
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        handleInput(key);
    }

    camera_.release();

    profiler_.printSummary();
    if (!opts_.csvPath.empty())
        profiler_.writeCsv(opts_.csvPath);
}
