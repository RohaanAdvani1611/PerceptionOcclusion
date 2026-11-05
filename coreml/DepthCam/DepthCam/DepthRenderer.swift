import Accelerate
import MetalKit
import SwiftUI

/// The newest depth image and its value range. Written on the camera queue, read when drawing.
nonisolated final class DepthFrameStore: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: (buffer: CVPixelBuffer, low: Float, high: Float)?
    private var floats = [Float](repeating: 0, count: 256 * 256)

    /// Finds the frame's min and max, so each frame fills the whole colour range.
    func put(_ depth: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        let width = CVPixelBufferGetWidth(depth), height = CVPixelBufferGetHeight(depth)
        if floats.count != width * height { floats = [Float](repeating: 0, count: width * height) }
        var low: Float = 0, high: Float = 0
        floats.withUnsafeMutableBufferPointer { out in
            var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(depth), height: vImagePixelCount(height),
                                    width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(depth))
            var dst = vImage_Buffer(data: out.baseAddress, height: vImagePixelCount(height),
                                    width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
            vDSP_minv(out.baseAddress!, 1, &low, vDSP_Length(out.count))
            vDSP_maxv(out.baseAddress!, 1, &high, vDSP_Length(out.count))
        }
        CVPixelBufferUnlockBaseAddress(depth, .readOnly)
        lock.withLock { latest = (depth, low, high) }
    }

    func take() -> (buffer: CVPixelBuffer, low: Float, high: Float)? {
        lock.withLock {
            defer { latest = nil }
            return latest
        }
    }
}

/// Draws the depth image with the inferno colormap, at the display's refresh rate.
final class DepthRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let frames = DepthFrameStore()
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var texture: MTLTexture?
    private var range: (low: Float, high: Float) = (0, 1)

    init?(pixelFormat: MTLPixelFormat = .bgra8Unorm) {
        // Compiled at launch, so building the app doesn't need Xcode's separate Metal Toolchain
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.shaderSource, options: nil) else { return nil }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "depthVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "depthFragment")
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        if let frame = frames.take() { upload(frame.buffer); range = (frame.low, frame.high) }
        guard let texture, let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let commands = commandQueue.makeCommandBuffer(),
              let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return }
        var depthRange = SIMD2<Float>(range.low, range.high)
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentBytes(&depthRange, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.present(drawable)
        commands.commit()
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float2 uv;
    };

    // One triangle that covers the whole view; no vertex buffer needed
    vertex VertexOut depthVertex(uint id [[vertex_id]]) {
        float2 p = float2((id << 1) & 2, id & 2);
        VertexOut out;
        out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
        out.uv = float2(p.x, 1.0 - p.y);
        return out;
    }

    // Polynomial fit of matplotlib's inferno colormap, the one the Python comparison images use
    static float3 inferno(float t) {
        const float3 c0 = float3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
        const float3 c1 = float3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
        const float3 c2 = float3(11.60249308247187, -3.972853965665698, -15.9423941062914);
        const float3 c3 = float3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
        const float3 c4 = float3(77.162935699427, -33.40235894210092, -81.80730925738993);
        const float3 c5 = float3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
        const float3 c6 = float3(25.13112622477341, -12.24266895238567, -23.07032500287172);
        return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
    }

    // Larger inverse depth means closer, so near things come out bright.
    // range.x and range.y are the frame's lowest and highest depth values.
    fragment float4 depthFragment(VertexOut in [[stage_in]],
                                  texture2d<float> depth [[texture(0)]],
                                  constant float2 &range [[buffer(0)]]) {
        constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
        float d = depth.sample(linearSampler, in.uv).r;
        float t = saturate((d - range.x) / max(range.y - range.x, 1e-3));
        return float4(inferno(t), 1.0);
    }
    """

    /// Copies the 256x256 Float16 depth image (128 KB) into a Metal texture.
    private func upload(_ buffer: CVPixelBuffer) {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        if texture?.width != width || texture?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: width,
                                                                      height: height, mipmapped: false)
            texture = device.makeTexture(descriptor: descriptor)
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        texture?.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                         withBytes: CVPixelBufferGetBaseAddress(buffer)!,
                         bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
}

#if os(macOS)
struct DepthView: NSViewRepresentable {
    let renderer: DepthRenderer

    func makeNSView(context: Context) -> MTKView { makeView() }
    func updateNSView(_ view: MTKView, context: Context) {}
}
#else
struct DepthView: UIViewRepresentable {
    let renderer: DepthRenderer

    func makeUIView(context: Context) -> MTKView { makeView() }
    func updateUIView(_ view: MTKView, context: Context) {}
}
#endif

extension DepthView {
    fileprivate func makeView() -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.delegate = renderer
        return view
    }
}
