import Accelerate
import CoreVideo

/// Stretches a BGRA camera frame to the model's 256x256 input with vImage.
///
/// Output buffers come from a pool, so they are reused rather than allocated per frame, and
/// are IOSurface-backed, so Core ML can hand them to the Neural Engine without a copy.
/// vImage resamples with a Lanczos filter, not bilinear like the C++ pipeline's cv::resize.
nonisolated final class FrameResizer {
    let size: Int
    private var pool: CVPixelBufferPool?

    init(size: Int = 256) {
        self.size = size
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: size,
            kCVPixelBufferHeightKey as String: size,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
    }

    func resize(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess,
              let destination else { return nil }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        var src = Self.vImageBuffer(source)
        var dst = Self.vImageBuffer(destination)
        // ARGB8888 scaling works for any 4-channel, 8-bit layout, including BGRA
        let error = vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
        return error == kvImageNoError ? destination : nil
    }

    private static func vImageBuffer(_ buffer: CVPixelBuffer) -> vImage_Buffer {
        vImage_Buffer(data: CVPixelBufferGetBaseAddress(buffer),
                      height: vImagePixelCount(CVPixelBufferGetHeight(buffer)),
                      width: vImagePixelCount(CVPixelBufferGetWidth(buffer)),
                      rowBytes: CVPixelBufferGetBytesPerRow(buffer))
    }
}
