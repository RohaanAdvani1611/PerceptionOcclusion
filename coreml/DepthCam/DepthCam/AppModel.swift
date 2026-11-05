import Foundation
import Observation

/// The app's state for the UI: which model is loaded, live timings, errors.
@Observable
final class AppModel {
    var variant: ModelVariant = .fp16
    var compute: ComputeChoice = .cpuAndNeuralEngine
    var stats = LiveStats()
    var loadInfo: LoadInfo?
    var isLoading = false
    var errorMessage: String?
    var planText: String?
    var isBenchmarking = false
    var benchmarkStatus: String?

    let renderer = DepthRenderer()
    private let camera = CameraManager()
    private let pipeline: DepthPipeline

    init() {
        pipeline = DepthPipeline(queue: camera.queue, frames: renderer?.frames ?? DepthFrameStore())
        camera.onFrame = { [pipeline] frame in pipeline.process(frame) }
        // AppModel lives as long as the app, so these closures can hold it strongly
        pipeline.onStats = { stats in
            Task { @MainActor in self.stats = stats }
        }
    }

    func start() async {
        reload()
        do {
            try await camera.start()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Loads the selected model and compute units. Loading can take a moment, since the
    /// first load for the Neural Engine compiles the model for it.
    func reload() {
        isLoading = true
        stats = LiveStats()
        planText = nil
        pipeline.load(variant, compute) { result in
            Task { @MainActor in
                self.isLoading = false
                switch result {
                case .success(let info):
                    self.loadInfo = info
                    self.errorMessage = nil
                    self.planText = (try? await ComputePlanSummary.load(info.compiledURL, compute: info.compute))?.text
                case .failure(let error):
                    self.loadInfo = nil
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    /// Pauses the live view, times all 16 combinations, then resumes.
    func runBenchmark() {
        isBenchmarking = true
        benchmarkStatus = "Starting…"
        pipeline.unload {
            Task { @MainActor in
                do {
                    let url = try await BenchmarkRunner().run { status in
                        Task { @MainActor in self.benchmarkStatus = status }
                    }
                    self.benchmarkStatus = "Saved \(url.path)"
                } catch {
                    self.benchmarkStatus = "Benchmark failed: \(error.localizedDescription)"
                }
                self.isBenchmarking = false
                self.reload()
            }
        }
    }
}
