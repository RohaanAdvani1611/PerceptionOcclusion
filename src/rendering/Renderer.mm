#import "rendering/Renderer.hpp"
#import <Metal/Metal.h>
#import <simd/simd.h>
#include <cmath>
#include <vector>
#include <iostream>

// ── Shared data types (must match vertex.metal layout) ───────────────────────

struct Vertex {
    float position[3]; // offset  0, 12 bytes
    float normal[3];   // offset 12, 12 bytes
};                     // stride 24

// Must match the Uniforms struct in the Metal shaders
struct Uniforms {
    simd_float4x4 mvp;
    simd_float4   color;
};

// ── Sphere generation ─────────────────────────────────────────────────────────

static void generateSphere(float radius, int rings, int sectors,
                            std::vector<Vertex>&   verts,
                            std::vector<uint32_t>& idxs) {
    verts.clear();
    idxs.clear();
    for (int r = 0; r <= rings; ++r) {
        float phi = M_PI * r / rings;
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * M_PI * s / sectors;
            float x = sinf(phi) * cosf(theta);
            float y = cosf(phi);
            float z = sinf(phi) * sinf(theta);
            verts.push_back({{radius * x, radius * y, radius * z}, {x, y, z}});
        }
    }
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t a = r * (sectors + 1) + s;
            uint32_t b = a + 1;
            uint32_t c = (r + 1) * (sectors + 1) + s;
            uint32_t d = c + 1;
            idxs.insert(idxs.end(), {a, c, b, b, c, d});
        }
    }
}

// ── Matrix helpers ────────────────────────────────────────────────────────────

// Standard perspective for Metal NDC depth [0, 1], camera looking down -Z (Y up).
static simd_float4x4 makePerspective(float fovY, float aspect, float near, float far) {
    float ys = 1.0f / tanf(fovY * 0.5f);
    float xs = ys / aspect;
    float zs = far / (near - far);
    simd_float4x4 m = {};
    m.columns[0] = { xs,  0,         0,  0 };
    m.columns[1] = {  0, ys,         0,  0 };
    m.columns[2] = {  0,  0,        zs, -1 };
    m.columns[3] = {  0,  0, near * zs,  0 };
    return m;
}

static simd_float4x4 makeTranslation(float x, float y, float z) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3] = { x, y, z, 1 };
    return m;
}

// ── Renderer ──────────────────────────────────────────────────────────────────

Renderer::Renderer(MetalContext& ctx) : ctx_(ctx) {}

void Renderer::moveSphere(float dx, float dy, float dz) {
    sphereX_ += dx;
    sphereY_ += dy;
    sphereZ_ += dz;
}

Renderer::~Renderer() {
    auto release = [](void*& p) { if (p) { CFBridgingRelease(p); p = nullptr; } };
    release(metalDepthTexture_);
    release(colorTexture_);
    release(indexBuffer_);
    release(vertexBuffer_);
    release(depthStencilState_);
    release(pipelineState_);
}

bool Renderer::init(int width, int height, const CameraIntrinsics& intrinsics) {
    width_      = width;
    height_     = height;
    intrinsics_ = intrinsics;

    id<MTLDevice>  dev = (__bridge id<MTLDevice>)ctx_.device();
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)ctx_.shaderLibrary();
    if (!dev || !lib) return false;

    // ── Vertex descriptor (matches Vertex struct layout) ─────────────────────
    MTLVertexDescriptor* vd     = [MTLVertexDescriptor new];
    vd.attributes[0].format     = MTLVertexFormatFloat3;
    vd.attributes[0].offset     = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format     = MTLVertexFormatFloat3;
    vd.attributes[1].offset     = 12;
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride        = 24;

    // ── Render pipeline ───────────────────────────────────────────────────────
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction               = [lib newFunctionWithName:@"vertex_main"];
    pd.fragmentFunction             = [lib newFunctionWithName:@"fragment_main"];
    pd.vertexDescriptor             = vd;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.depthAttachmentPixelFormat      = MTLPixelFormatDepth32Float;

    NSError* err = nil;
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) {
        std::cerr << "Renderer: pipeline error: " << err.localizedDescription.UTF8String << "\n";
        return false;
    }
    pipelineState_ = (void*)CFBridgingRetain(ps);

    // ── Depth/stencil state ───────────────────────────────────────────────────
    MTLDepthStencilDescriptor* dsd = [MTLDepthStencilDescriptor new];
    dsd.depthCompareFunction = MTLCompareFunctionLess;
    dsd.depthWriteEnabled    = YES;
    depthStencilState_ = (void*)CFBridgingRetain([dev newDepthStencilStateWithDescriptor:dsd]);

    // ── Sphere mesh ───────────────────────────────────────────────────────────
    std::vector<Vertex>   verts;
    std::vector<uint32_t> idxs;
    generateSphere(0.12f, 24, 24, verts, idxs);
    indexCount_ = static_cast<int>(idxs.size());

    vertexBuffer_ = (void*)CFBridgingRetain(
        [dev newBufferWithBytes:verts.data()
                        length:verts.size() * sizeof(Vertex)
                       options:MTLResourceStorageModeShared]);
    indexBuffer_  = (void*)CFBridgingRetain(
        [dev newBufferWithBytes:idxs.data()
                        length:idxs.size() * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared]);

    // ── Render target textures ────────────────────────────────────────────────
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.width       = width;
    td.height      = height;
    td.storageMode = MTLStorageModeShared; // CPU-readable; free on M-series unified memory

    td.pixelFormat = MTLPixelFormatBGRA8Unorm;
    td.usage       = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    colorTexture_  = (void*)CFBridgingRetain([dev newTextureWithDescriptor:td]);

    td.pixelFormat      = MTLPixelFormatDepth32Float;
    td.usage            = MTLTextureUsageRenderTarget;
    metalDepthTexture_  = (void*)CFBridgingRetain([dev newTextureWithDescriptor:td]);

    ready_ = true;
    std::cout << "Renderer: initialized " << width << "x" << height << "\n";
    return true;
}

cv::Mat Renderer::render() {
    if (!ready_) return {};

    id<MTLCommandQueue>        cq  = (__bridge id<MTLCommandQueue>)ctx_.commandQueue();
    id<MTLRenderPipelineState> ps  = (__bridge id<MTLRenderPipelineState>)pipelineState_;
    id<MTLDepthStencilState>   dss = (__bridge id<MTLDepthStencilState>)depthStencilState_;
    id<MTLTexture>             col = (__bridge id<MTLTexture>)colorTexture_;
    id<MTLTexture>             dep = (__bridge id<MTLTexture>)metalDepthTexture_;
    id<MTLBuffer>              vb  = (__bridge id<MTLBuffer>)vertexBuffer_;
    id<MTLBuffer>              ib  = (__bridge id<MTLBuffer>)indexBuffer_;

    // ── MVP ───────────────────────────────────────────────────────────────────
    // Derive vertical FOV from camera intrinsics; sphere placed 1.5 m in front, Y-up camera space
    float fovY   = 2.0f * atanf(static_cast<float>(height_) / (2.0f * intrinsics_.fy));
    float aspect = static_cast<float>(width_) / static_cast<float>(height_);
    simd_float4x4 proj  = makePerspective(fovY, aspect, 0.1f, 20.0f);
    simd_float4x4 model = makeTranslation(sphereX_, sphereY_, sphereZ_);

    Uniforms u;
    u.mvp   = simd_mul(proj, model);
    u.color = { 1.0f, 0.45f, 0.1f, 1.0f }; // warm orange

    // ── Render pass ───────────────────────────────────────────────────────────
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = col;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 0); // transparent bg
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture         = dep;
    rpd.depthAttachment.loadAction      = MTLLoadActionClear;
    rpd.depthAttachment.clearDepth      = 1.0;
    rpd.depthAttachment.storeAction     = MTLStoreActionDontCare;

    id<MTLCommandBuffer>        cb  = [cq commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];

    [enc setRenderPipelineState:ps];
    [enc setDepthStencilState:dss];
    [enc setVertexBuffer:vb offset:0 atIndex:0];
    [enc setVertexBytes:&u  length:sizeof(u) atIndex:1];
    [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                    indexCount:indexCount_
                     indexType:MTLIndexTypeUInt32
                   indexBuffer:ib
             indexBufferOffset:0];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];

    // ── Readback color texture → cv::Mat BGRA ────────────────────────────────
    cv::Mat result(height_, width_, CV_8UC4);
    [col getBytes:result.data
      bytesPerRow:width_ * 4
       fromRegion:MTLRegionMake2D(0, 0, width_, height_)
      mipmapLevel:0];
    return result;
}
