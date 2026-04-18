#pragma once
#include <string>

// Pure C++ header — no Obj-C types exposed.
// Metal objects are stored as void* and cast inside MetalContext.mm.
class MetalContext {
public:
    MetalContext() = default;
    ~MetalContext();

    // shaderDir: directory containing vertex.metal + fragment_occlusion.metal
    bool init(const std::string& shaderDir);
    bool isValid() const { return device_ != nullptr; }

    void* device()        const { return device_; }
    void* commandQueue()  const { return commandQueue_; }
    void* shaderLibrary() const { return library_; }

private:
    void* device_       = nullptr;
    void* commandQueue_ = nullptr;
    void* library_      = nullptr;
};
