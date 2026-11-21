// Compiled with ARC (see CMakeLists.txt), so Objective-C objects in Impl are released automatically.
#include "perception/CoreMLDepthEstimator.hpp"
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace {
constexpr int kInput = 256;  // the model's fixed input size
}

struct CoreMLDepthEstimator::Impl {
    std::string modelPath;
    MLModel* model = nil;
    CVPixelBufferRef input = nullptr;  // reused every frame
    cv::Mat resized;

    ~Impl() { if (input) CVPixelBufferRelease(input); }
};

CoreMLDepthEstimator::CoreMLDepthEstimator(const std::string& modelPath)
    : impl_(std::make_unique<Impl>()) {
    impl_->modelPath = modelPath;
}

CoreMLDepthEstimator::~CoreMLDepthEstimator() = default;

bool CoreMLDepthEstimator::isLoaded() const { return impl_->model != nil; }

bool CoreMLDepthEstimator::load() {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:@(impl_->modelPath.c_str())];
        if (![[NSFileManager defaultManager] fileExistsAtPath:url.path]) {
            std::cerr << "CoreMLDepthEstimator: no model at " << impl_->modelPath << "\n";
            return false;
        }

        // An .mlpackage must be compiled to .mlmodelc before loading. This takes about 0.1 s.
        // The first load on the Neural Engine also compiles for it (about 1 s), and the
        // system caches that result for later runs.
        NSError* error = nil;
        NSURL* compiled = url;
        if ([url.pathExtension isEqualToString:@"mlpackage"]) {
            compiled = [MLModel compileModelAtURL:url error:&error];
            if (!compiled) {
                std::cerr << "CoreMLDepthEstimator: compile failed: " << error.localizedDescription.UTF8String << "\n";
                return false;
            }
        }

        MLModelConfiguration* config = [MLModelConfiguration new];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        impl_->model = [MLModel modelWithContentsOfURL:compiled configuration:config error:&error];
        if (!impl_->model) {
            std::cerr << "CoreMLDepthEstimator: load failed: " << error.localizedDescription.UTF8String << "\n";
            return false;
        }

        // IOSurface-backed, so the Neural Engine can read it without an extra copy
        NSDictionary* attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}};
        if (CVPixelBufferCreate(kCFAllocatorDefault, kInput, kInput, kCVPixelFormatType_32BGRA,
                                (__bridge CFDictionaryRef)attrs, &impl_->input) != kCVReturnSuccess) {
            std::cerr << "CoreMLDepthEstimator: could not create the input buffer\n";
            impl_->model = nil;
            return false;
        }
    }
    std::cout << "CoreMLDepthEstimator: model loaded  input=" << kInput << "x" << kInput
              << "  compute=CPU+Neural Engine  " << impl_->modelPath << "\n";
    return true;
}

cv::Mat CoreMLDepthEstimator::infer(const cv::Mat& bgr) {
    if (!impl_->model) return {};

    // Same bilinear resize as the frames the accuracy evaluation used
    cv::resize(bgr, impl_->resized, cv::Size(kInput, kInput), 0, 0, cv::INTER_LINEAR);

    CVPixelBufferLockBaseAddress(impl_->input, 0);
    cv::Mat inputView(kInput, kInput, CV_8UC4, CVPixelBufferGetBaseAddress(impl_->input),
                      CVPixelBufferGetBytesPerRow(impl_->input));
    cv::cvtColor(impl_->resized, inputView, cv::COLOR_BGR2BGRA);  // writes straight into the buffer
    CVPixelBufferUnlockBaseAddress(impl_->input, 0);

    cv::Mat raw;
    @autoreleasepool {
        NSError* error = nil;
        MLFeatureValue* value = [MLFeatureValue featureValueWithPixelBuffer:impl_->input];
        MLDictionaryFeatureProvider* features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"image": value} error:&error];
        id<MLFeatureProvider> output = features ? [impl_->model predictionFromFeatures:features error:&error] : nil;
        CVPixelBufferRef depth = [output featureValueForName:@"depth"].imageBufferValue;
        if (!depth) {
            std::cerr << "CoreMLDepthEstimator: prediction failed: "
                      << (error ? error.localizedDescription.UTF8String : "no depth output") << "\n";
            return {};
        }

        // Float16 grayscale → float32, copied out before the output buffer is released
        CVPixelBufferLockBaseAddress(depth, kCVPixelBufferLock_ReadOnly);
        cv::Mat half(static_cast<int>(CVPixelBufferGetHeight(depth)), static_cast<int>(CVPixelBufferGetWidth(depth)),
                     CV_16FC1, CVPixelBufferGetBaseAddress(depth), CVPixelBufferGetBytesPerRow(depth));
        half.convertTo(raw, CV_32F);
        CVPixelBufferUnlockBaseAddress(depth, kCVPixelBufferLock_ReadOnly);
    }

    return normalizeDepth(raw, bgr.size());
}
