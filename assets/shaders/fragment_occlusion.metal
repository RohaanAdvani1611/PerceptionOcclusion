// VertexOut and Uniforms are defined in vertex.metal.
// Both files are concatenated into one compilation unit by MetalContext.

fragment float4 fragment_main(VertexOut          in [[stage_in]],
                               constant Uniforms&  u  [[buffer(1)]]) {
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float  diffuse  = max(dot(normalize(in.normal), lightDir), 0.0f);
    float3 shaded   = u.color.rgb * (0.25 + 0.75 * diffuse);
    return float4(shaded, 1.0);
}
