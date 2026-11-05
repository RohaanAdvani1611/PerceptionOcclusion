import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

/// Times every model version under every compute-unit setting on the 100 held-out frames,
/// and writes one CSV row per combination to the app's Documents folder.
///
/// Per combination: load twice (the first load for the Neural Engine may compile the model),
/// 30 untimed warm-up predictions, 500 timed ones, then a cooldown so heat from one run
/// doesn't slow the next. Thermal state is logged before and after each run.
nonisolated struct BenchmarkRunner {
    var warmup = 30
    var timed = 500
    var cooldownSeconds = 10.0

    enum BenchmarkError: LocalizedError {
        case noFrames
        var errorDescription: String? { "No bench_*.png frames in the app. Run export_bench_frames.py." }
    }

    /// Runs off the main thread. `progress` gets a short status line before each combination.
    @concurrent
    func run(progress: @escaping @Sendable (String) -> Void) async throws -> URL {
        let frames = try Self.loadFrames()
        guard !frames.isEmpty else { throw BenchmarkError.noFrames }
        let combinations = ModelVariant.allCases.flatMap { v in ComputeChoice.allCases.map { (v, $0) } }

        var lines = [Self.header]
        for (index, (variant, compute)) in combinations.enumerated() {
            progress("Run \(index + 1)/\(combinations.count): \(variant.label), \(compute.label)")
            lines.append(try await measure(variant, compute, frames: frames))
            if index < combinations.count - 1 {
                progress("Cooling down \(Int(cooldownSeconds)) s after run \(index + 1)/\(combinations.count)")
                try await Task.sleep(for: .seconds(cooldownSeconds))
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = documents.appendingPathComponent("benchmark_\(Self.deviceName)_\(stamp).csv")
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func measure(_ variant: ModelVariant, _ compute: ComputeChoice,
                         frames: [CVPixelBuffer]) async throws -> String {
        let thermalBefore = Self.thermal()
        let first = try ModelRunner(variant: variant, compute: compute)
        let firstLoad = first.loadMilliseconds
        let plan = try await ComputePlanSummary.load(first.compiledURL, compute: compute)
        let runner = try ModelRunner(variant: variant, compute: compute)  // second, cached load

        for i in 0..<warmup { _ = try runner.predict(frames[i % frames.count]) }
        var times: [Double] = []
        times.reserveCapacity(timed)
        for i in 0..<timed {
            let start = milliseconds()
            _ = try runner.predict(frames[i % frames.count])
            times.append(milliseconds() - start)
        }
        let mean = times.reduce(0, +) / Double(times.count)

        let fields: [String] = [
            Self.deviceName, Self.osVersion, variant.rawValue, compute.rawValue,
            String(format: "%.2f", runner.compiledMegabytes),
            String(format: "%.1f", firstLoad), String(format: "%.1f", runner.loadMilliseconds),
            String(format: "%.3f", percentile(times, 0.5)), String(format: "%.3f", percentile(times, 0.9)),
            String(format: "%.3f", percentile(times, 0.99)), String(format: "%.3f", mean),
            String(plan.operations),
            String(format: "%.1f", plan.percentOps("ANE")), String(format: "%.1f", plan.percentCost("ANE")),
            String(format: "%.1f", plan.percentOps("GPU")), String(format: "%.1f", plan.percentOps("CPU")),
            thermalBefore, Self.thermal(), String(ProcessInfo.processInfo.isLowPowerModeEnabled),
        ]
        return fields.joined(separator: ",")
    }

    static let header = "device,os,variant,compute,compiled_mb,load_first_ms,load_second_ms,"
        + "p50_ms,p90_ms,p99_ms,mean_ms,plan_ops,ane_ops_pct,ane_cost_pct,gpu_ops_pct,cpu_ops_pct,"
        + "thermal_before,thermal_after,low_power_mode"

    /// "Mac14,15" for the MacBook Air M2, "iPhone14,7" for the iPhone 14.
    static var deviceName: String {
        #if os(iOS)
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        #else
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
        #endif
    }

    static var osVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// The bundled bench_*.png frames as 256x256 BGRA, IOSurface-backed pixel buffers.
    static func loadFrames() throws -> [CVPixelBuffer] {
        let urls = (Bundle.main.urls(forResourcesWithExtension: "png", subdirectory: nil) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("bench_") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return urls.compactMap { url in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
            return pixelBuffer(from: image)
        }
    }

    private static func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, image.width, image.height, kCVPixelFormatType_32BGRA,
                            attributes as CFDictionary, &buffer)
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        // BGRA in memory is 32-bit little-endian with alpha first. Drawing in the image's own
        // colour space avoids any colour conversion of the pixels.
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: image.width, height: image.height,
                                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        context?.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return buffer
    }
}
