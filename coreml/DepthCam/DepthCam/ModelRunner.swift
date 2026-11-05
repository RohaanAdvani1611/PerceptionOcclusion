import CoreML
import Foundation

/// The four compressed versions built by coreml/compress.py, bundled from DepthCam/Models/.
nonisolated enum ModelVariant: String, CaseIterable, Identifiable, Sendable {
    case fp16, w8, p6, w8a8

    var id: String { rawValue }
    var resourceName: String { "midas_small_\(rawValue)" }
    var label: String {
        switch self {
        case .fp16: "FP16"
        case .w8: "8-bit weights"
        case .p6: "6-bit palette"
        case .w8a8: "W8A8"
        }
    }
}

/// Which processors Core ML may use. It's permission, not a command: ops can still fall back to the CPU.
nonisolated enum ComputeChoice: String, CaseIterable, Identifiable, Sendable {
    case all, cpuAndNeuralEngine, cpuAndGPU, cpuOnly

    var id: String { rawValue }
    var units: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuAndGPU: .cpuAndGPU
        case .cpuOnly: .cpuOnly
        }
    }
    var label: String {
        switch self {
        case .all: "All"
        case .cpuAndNeuralEngine: "CPU + Neural Engine"
        case .cpuAndGPU: "CPU + GPU"
        case .cpuOnly: "CPU only"
        }
    }
}

enum ModelError: LocalizedError {
    case missing(String), noDepthOutput

    var errorDescription: String? {
        switch self {
        case .missing(let name): "\(name).mlmodelc is not in the app. Copy the .mlpackage files into DepthCam/Models/."
        case .noDepthOutput: "The model returned no depth image."
        }
    }
}

/// Milliseconds on a monotonic clock, for timing.
nonisolated func milliseconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000
}

/// One loaded model version with a fixed compute-unit setting.
nonisolated final class ModelRunner {
    let variant: ModelVariant
    let compute: ComputeChoice
    let compiledURL: URL
    let loadMilliseconds: Double
    let compiledMegabytes: Double
    private let model: MLModel

    init(variant: ModelVariant, compute: ComputeChoice) throws {
        // Xcode compiles each bundled .mlpackage into a ready-to-run .mlmodelc
        guard let url = Bundle.main.url(forResource: variant.resourceName, withExtension: "mlmodelc") else {
            throw ModelError.missing(variant.resourceName)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = compute.units

        let start = milliseconds()
        model = try MLModel(contentsOf: url, configuration: configuration)
        loadMilliseconds = milliseconds() - start

        self.variant = variant
        self.compute = compute
        compiledURL = url
        compiledMegabytes = Self.directoryMegabytes(url)
    }

    /// 256x256 BGRA in, 256x256 Float16 relative inverse depth out (larger is closer).
    func predict(_ image: CVPixelBuffer) throws -> CVPixelBuffer {
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: image)])
        let output = try model.prediction(from: input)
        guard let depth = output.featureValue(for: "depth")?.imageBufferValue else { throw ModelError.noDepthOutput }
        return depth
    }

    private static func directoryMegabytes(_ url: URL) -> Double {
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var bytes = 0
        while let file = files?.nextObject() as? URL {
            bytes += (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return Double(bytes) / 1_000_000
    }
}
