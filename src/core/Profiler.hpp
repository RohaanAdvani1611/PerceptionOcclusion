#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Per-stage frame timing. Each frame, record() any subset of stages, then call
// endFrame(). The first `warmupFrames` frames are excluded from the run summary
// so ONNX Runtime / Metal warm-up doesn't skew the statistics.
class Profiler {
public:
    using Clock = std::chrono::steady_clock;

    Profiler(std::vector<std::string> stages, int warmupFrames)
        : names_(std::move(stages)), warmup_(warmupFrames),
          current_(names_.size(), kMissing),
          windowSum_(names_.size(), 0.0), windowCount_(names_.size(), 0) {}

    void record(const std::string& stage, double ms) {
        for (size_t i = 0; i < names_.size(); ++i)
            if (names_[i] == stage) { current_[i] = ms; return; }
    }

    void endFrame() {
        auto now = Clock::now();
        for (size_t i = 0; i < names_.size(); ++i) {
            if (std::isnan(current_[i])) continue;
            windowSum_[i] += current_[i];
            ++windowCount_[i];
        }
        ++windowFrames_;

        if (framesSeen_ == warmup_) measureStart_ = now;
        if (framesSeen_ >= warmup_) {
            rows_.push_back(current_);
            measureEnd_ = now;
        }
        ++framesSeen_;
        std::fill(current_.begin(), current_.end(), kMissing);
    }

    int measuredFrames() const { return static_cast<int>(rows_.size()); }

    // Prints one line of per-stage means since the last call, then resets the window.
    void printWindow(double elapsedSec) {
        std::cout << std::fixed << std::setprecision(1)
                  << "FPS: " << windowFrames_ / elapsedSec;
        for (size_t i = 0; i < names_.size(); ++i) {
            if (windowCount_[i] == 0) continue;
            std::cout << "  " << names_[i] << ": " << windowSum_[i] / windowCount_[i] << "ms";
        }
        std::cout << "\n";
        std::fill(windowSum_.begin(), windowSum_.end(), 0.0);
        std::fill(windowCount_.begin(), windowCount_.end(), 0);
        windowFrames_ = 0;
    }

    // ── Run summary ───────────────────────────────────────────────────────────

    void printSummary() const {
        if (rows_.size() < 2) {
            std::cout << "Profiler: not enough frames past warm-up for a summary\n";
            return;
        }

        std::cout << "\n" << std::fixed << std::setprecision(2)
                  << std::left << std::setw(12) << "stage (ms)" << std::right
                  << std::setw(9) << "mean" << std::setw(9) << "p50"
                  << std::setw(9) << "p95"  << std::setw(9) << "p99"
                  << std::setw(9) << "max"  << "\n";

        for (size_t i = 0; i < names_.size(); ++i) {
            std::vector<double> v;
            for (const auto& row : rows_)
                if (!std::isnan(row[i])) v.push_back(row[i]);
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());

            double mean = 0.0;
            for (double x : v) mean += x;
            mean /= v.size();

            std::cout << std::left << std::setw(12) << names_[i] << std::right
                      << std::setw(9) << mean
                      << std::setw(9) << percentile(v, 0.50)
                      << std::setw(9) << percentile(v, 0.95)
                      << std::setw(9) << percentile(v, 0.99)
                      << std::setw(9) << v.back() << "\n";
        }

        // FPS spans endFrame() of the last warm-up frame to endFrame() of the last frame
        double sec = std::chrono::duration<double>(measureEnd_ - measureStart_).count();
        std::cout << std::setprecision(1)
                  << "FPS: " << (rows_.size() - 1) / sec << " over " << rows_.size()
                  << " frames (" << warmup_ << " warm-up skipped)\n";
    }

    bool writeCsv(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            std::cerr << "Profiler: cannot write " << path << "\n";
            return false;
        }
        out << "frame";
        for (const auto& n : names_) out << "," << n << "_ms";
        out << "\n" << std::fixed << std::setprecision(3);
        for (size_t f = 0; f < rows_.size(); ++f) {
            out << f;
            for (double ms : rows_[f]) {
                out << ",";
                if (!std::isnan(ms)) out << ms;
            }
            out << "\n";
        }
        std::cout << "Profiler: wrote " << rows_.size() << " frames to " << path << "\n";
        return true;
    }

private:
    static constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

    // Nearest-rank percentile on a sorted vector
    static double percentile(const std::vector<double>& sorted, double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * sorted.size()));
        return sorted[std::min(sorted.size() - 1, idx > 0 ? idx - 1 : 0)];
    }

    std::vector<std::string>         names_;
    int                              warmup_;
    int                              framesSeen_   = 0;
    int                              windowFrames_ = 0;
    std::vector<double>              current_;
    std::vector<double>              windowSum_;
    std::vector<int>                 windowCount_;
    std::vector<std::vector<double>> rows_;
    Clock::time_point                measureStart_ {};
    Clock::time_point                measureEnd_   {};
};
