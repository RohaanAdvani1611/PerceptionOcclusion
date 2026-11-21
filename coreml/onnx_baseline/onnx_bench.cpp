// Times the C++ pipeline's ONNX model on the same 100 frames as DepthCam's benchmark mode.
//
// Only session.Run is timed, so the number compares with Core ML's prediction time.
// Session settings match DepthEstimator.cpp: CPU, 4 intra-op threads, full graph optimization.
// Input preparation matches it too (BGR -> RGB, /255, NCHW; the graph normalizes itself),
// minus the resize, since the bench frames are already 256x256.
//
// Usage: onnx_bench [--threads N] [--warmup N] [--runs N] [--csv path]
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const fs::path kRepo = fs::path(__FILE__).parent_path().parent_path().parent_path();

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

// 256x256 BGR uint8 -> 1x3x256x256 RGB float in [0, 1], as DepthEstimator::preprocess does
static std::vector<float> toBlob(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    const int h = rgb.rows, w = rgb.cols;
    std::vector<float> blob(3 * h * w);
    std::vector<cv::Mat> planes(3);
    for (int c = 0; c < 3; ++c) planes[c] = cv::Mat(h, w, CV_32F, blob.data() + c * h * w);
    cv::split(rgb, planes);
    return blob;
}

int main(int argc, char** argv) {
    int threads = 4, warmup = 30, runs = 500;
    fs::path csv = kRepo / "inference" / "coreml" / "onnx_bench.csv";
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        if (flag == "--threads") threads = std::atoi(argv[i + 1]);
        else if (flag == "--warmup") warmup = std::atoi(argv[i + 1]);
        else if (flag == "--runs") runs = std::atoi(argv[i + 1]);
        else if (flag == "--csv") csv = argv[i + 1];
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(kRepo / "coreml" / "DepthCam" / "DepthCam" / "BenchFrames")) {
        if (entry.path().filename().string().rfind("bench_", 0) == 0) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        std::cerr << "No bench frames. Run: python coreml/export_bench_frames.py\n";
        return 1;
    }
    std::vector<std::vector<float>> blobs;
    for (const auto& p : paths) blobs.push_back(toBlob(cv::imread(p.string())));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_bench");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const fs::path model = kRepo / "assets" / "models" / "midas.onnx";

    const auto loadStart = Clock::now();
    Ort::Session session(env, model.c_str(), options);
    const double loadMs = ms(loadStart, Clock::now());

    Ort::AllocatorWithDefaultOptions allocator;
    const std::string inputName = session.GetInputNameAllocated(0, allocator).get();
    const std::string outputName = session.GetOutputNameAllocated(0, allocator).get();
    const char* inNames[] = {inputName.c_str()};
    const char* outNames[] = {outputName.c_str()};
    const std::array<int64_t, 4> shape{1, 3, 256, 256};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto run = [&](std::vector<float>& blob) {
        auto input = Ort::Value::CreateTensor<float>(memory, blob.data(), blob.size(), shape.data(), shape.size());
        return session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
    };

    for (int i = 0; i < warmup; ++i) run(blobs[i % blobs.size()]);
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const auto start = Clock::now();
        run(blobs[i % blobs.size()]);
        times.push_back(ms(start, Clock::now()));
    }
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double p50 = percentile(times, 0.5), p90 = percentile(times, 0.9), p99 = percentile(times, 0.99);

    std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString() << ", CPU, " << threads << " threads, "
              << blobs.size() << " frames, " << warmup << " warm-up + " << runs << " timed runs\n"
              << "load " << loadMs << " ms\n"
              << "session.Run p50 " << p50 << " ms, p90 " << p90 << " ms, p99 " << p99 << " ms, mean " << mean
              << " ms\n";

    fs::create_directories(csv.parent_path());
    const bool newFile = !fs::exists(csv);
    std::ofstream out(csv, std::ios::app);
    if (newFile) out << "ort_version,threads,frames,warmup,runs,load_ms,p50_ms,p90_ms,p99_ms,mean_ms\n";
    out << OrtGetApiBase()->GetVersionString() << ',' << threads << ',' << blobs.size() << ',' << warmup << ','
        << runs << ',' << loadMs << ',' << p50 << ',' << p90 << ',' << p99 << ',' << mean << '\n';
    std::cout << "Appended to " << csv << "\n";
    return 0;
}
