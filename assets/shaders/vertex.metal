#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

// Shared with fragment_occlusion.metal (compiled as one unit)
struct VertexOut {
    float4 position [[position]];
    float3 normal;
};

struct Uniforms {
    float4x4 mvp;
    float4   color;
};

vertex VertexOut vertex_main(VertexIn           in [[stage_in]],
                              constant Uniforms&  u  [[buffer(1)]]) {
    VertexOut out;
    out.position = u.mvp * float4(in.position, 1.0);
    out.normal   = in.normal;
    return out;
}
