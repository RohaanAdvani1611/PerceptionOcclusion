// VertexOut and Uniforms are defined in vertex.metal.
// Both files are concatenated into one compilation unit by MetalContext.

fragment float4 fragment_background(float4 pos [[position]],
                                     texture2d<float> camera [[texture(0)]]) {
    return camera.read(uint2(pos.xy));
}

fragment float4 fragment_main(VertexOut          in [[stage_in]],
                               constant Uniforms&  u  [[buffer(1)]],
                               texture2d<float>    realDepth [[texture(1)]]) {
    // Real depth is inverse depth (bigger = closer), so convert the sphere's
    // distance to the same scale and hide this pixel if the real surface is closer.
    float sphereDepth = u.occlusionScale / in.viewDistance;
    if (realDepth.read(uint2(in.position.xy)).r > sphereDepth)
        discard_fragment();

    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float  diffuse  = max(dot(normalize(in.normal), lightDir), 0.0f);
    float3 shaded   = u.color.rgb * (0.25 + 0.75 * diffuse);
    return float4(shaded, 1.0);
}
