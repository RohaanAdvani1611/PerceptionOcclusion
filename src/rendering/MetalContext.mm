#import "rendering/MetalContext.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>

// Metal objects are stored as void* in the header.
// CFBridgingRetain/Release manage ref-counts across the ARC / void* boundary.

MetalContext::~MetalContext() {
    if (library_)      { CFBridgingRelease(library_);      library_      = nullptr; }
    if (commandQueue_) { CFBridgingRelease(commandQueue_); commandQueue_ = nullptr; }
    if (device_)       { CFBridgingRelease(device_);       device_       = nullptr; }
}

bool MetalContext::init(const std::string& shaderDir) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        std::cerr << "MetalContext: no Metal device found\n";
        return false;
    }
    device_       = (void*)CFBridgingRetain(dev);
    commandQueue_ = (void*)CFBridgingRetain([dev newCommandQueue]);

    auto readFile = [](const std::string& path) -> NSString* {
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        return [NSString stringWithContentsOfFile:nsPath
                                         encoding:NSUTF8StringEncoding
                                            error:nil];
    };

    NSString* vertSrc = readFile(shaderDir + "/vertex.metal");
    NSString* fragSrc = readFile(shaderDir + "/fragment_occlusion.metal");
    if (!vertSrc || !fragSrc) {
        std::cerr << "MetalContext: shader sources not found in " << shaderDir << "\n";
        return false;
    }

    // Concatenate so fragment shader can use VertexOut defined in vertex.metal
    NSString* combined = [vertSrc stringByAppendingString:fragSrc];
    NSError*  err      = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:combined options:nil error:&err];
    if (!lib) {
        std::cerr << "MetalContext: shader compile error: "
                  << err.localizedDescription.UTF8String << "\n";
        return false;
    }
    library_ = (void*)CFBridgingRetain(lib);

    std::cout << "MetalContext: " << dev.name.UTF8String << " ready\n";
    return true;
}
