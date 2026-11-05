import CoreVideo
import Foundation

nonisolated struct LiveStats: Sendable {
    var cameraWidth = 0
    var cameraHeight = 0
    var fps = 0.0
    var resizeP50 = 0.0
    var predictP50 = 0.0
    var predictP99 = 0.0
}

nonisolated struct LoadInfo: Sendable {
    var variant: ModelVariant
    var compute: ComputeChoice
    var loadMilliseconds: Double
    var compiledMegabytes: Double
    var compiledURL: URL
}

/// Camera frame -> 256x256 resize -> Core ML -> display, timing each step.
///
/// Everything runs on the camera's serial queue, so loading a model never overlaps a
/// prediction. Frames that arrive while a frame or a load is in progress are dropped.
nonisolated final class DepthPipeline: @unchecked Sendable {
    /// Called on the camera queue about twice a second.
    var onStats: (@Sendable (LiveStats) -> Void)?

    private let queue: DispatchQueue
    private let frames: DepthFrameStore
    private let resizer = FrameResizer()
    private var runner: ModelRunner?
    private var resizeTimes: [Double] = []
    private var predictTimes: [Double] = []
    private var finishTimes: [Double] = []
    private var lastReport = 0.0
    private var framesSinceLoad = 0
    private let window = 120
    /// The first predictions after a load are slow while the model warms up, so they're left
    /// out of the timings, as the benchmark leaves out its warm-up runs.
    private let warmupFrames = 10

    init(queue: DispatchQueue, frames: DepthFrameStore) {
        self.queue = queue
        self.frames = frames
    }

    func load(_ variant: ModelVariant, _ compute: ComputeChoice,
              completion: @escaping @Sendable (Result<LoadInfo, Error>) -> Void) {
        queue.async {
            self.runner = nil  // release the old model first, memory is tight on an 8 GB Mac
            self.framesSinceLoad = 0
            self.resizeTimes.removeAll()
            self.predictTimes.removeAll()
            self.finishTimes.removeAll()
            do {
                let runner = try ModelRunner(variant: variant, compute: compute)
                self.runner = runner
                completion(.success(LoadInfo(variant: variant, compute: compute,
                                             loadMilliseconds: runner.loadMilliseconds,
                                             compiledMegabytes: runner.compiledMegabytes,
                                             compiledURL: runner.compiledURL)))
            } catch {
                completion(.failure(error))
            }
        }
    }

    /// Releases the model, so live frames stop being processed (used during benchmarks).
    func unload(completion: @escaping @Sendable () -> Void) {
        queue.async {
            self.runner = nil
            completion()
        }
    }

    /// Called on the camera queue for each frame.
    func process(_ frame: CVPixelBuffer) {
        guard let runner else { return }
        let start = milliseconds()
        guard let input = resizer.resize(frame) else { return }
        let resized = milliseconds()
        guard let depth = try? runner.predict(input) else { return }
        let predicted = milliseconds()
        frames.put(depth)

        framesSinceLoad += 1
        guard framesSinceLoad > warmupFrames else { return }
        append(&resizeTimes, resized - start)
        append(&predictTimes, predicted - resized)
        finishTimes.append(predicted)
        finishTimes.removeAll { $0 < predicted - 1000 }
        if predicted - lastReport > 500 {
            lastReport = predicted
            onStats?(LiveStats(cameraWidth: CVPixelBufferGetWidth(frame),
                               cameraHeight: CVPixelBufferGetHeight(frame),
                               fps: Double(finishTimes.count),
                               resizeP50: percentile(resizeTimes, 0.5),
                               predictP50: percentile(predictTimes, 0.5),
                               predictP99: percentile(predictTimes, 0.99)))
        }
    }

    private func append(_ times: inout [Double], _ value: Double) {
        times.append(value)
        if times.count > window { times.removeFirst(times.count - window) }
    }
}

nonisolated func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(p * Double(sorted.count)))]
}
