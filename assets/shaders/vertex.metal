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
    float  viewDistance; // distance in front of the camera, metres
};

struct Uniforms {
    float4x4 mvp;
    float4   color;
    float    occlusionScale; // real depth value of a surface 1 metre away
};

vertex VertexOut vertex_main(VertexIn           in [[stage_in]],
                              constant Uniforms&  u  [[buffer(1)]]) {
    VertexOut out;
    out.position     = u.mvp * float4(in.position, 1.0);
    out.normal       = in.normal;
    out.viewDistance = out.position.w; // the perspective matrix sets clip w to -z_view
    return out;
}

// Oversized triangle covering the whole render target
vertex float4 vertex_background(uint id [[vertex_id]]) {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}
