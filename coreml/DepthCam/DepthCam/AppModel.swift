import Foundation
import Observation
#if os(iOS)
import UIKit
#endif

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
    var soakStatus: String?
    private var soakTimer: Timer?
    private var soakLines: [String] = []
    private var soakStart = Date()
    /// Held for the app's lifetime. Without it, macOS App Nap throttles DepthCam whenever another
    /// window is in front, which stopped the live model during the first power readings.
    private var liveActivity: NSObjectProtocol?

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
        liveActivity = ProcessInfo.processInfo.beginActivity(options: .userInitiated,
                                                             reason: "Live depth from the camera")
        #if os(iOS)
        // Keeps the phone from auto-locking, which would stop the camera during the soak test
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
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

    var isSoakRecording: Bool { soakTimer != nil }

    /// Records the live timings and thermal state once a second, to see whether heat slows
    /// the model down over a long run. Stopping saves soak_<device>_<time>.csv to Documents.
    func toggleSoakRecording() {
        if let timer = soakTimer {
            timer.invalidate()
            soakTimer = nil
            saveSoakLog()
            return
        }
        soakStart = Date()
        soakLines = ["seconds,variant,compute,fps,predict_p50_ms,predict_p99_ms,resize_p50_ms,thermal"]
        soakTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            Task { @MainActor in self.recordSoakSample() }
        }
    }

    private func recordSoakSample() {
        let seconds = Int(Date().timeIntervalSince(soakStart))
        soakLines.append(String(format: "%d,%@,%@,%.0f,%.3f,%.3f,%.3f,%@", seconds, variant.rawValue,
                                compute.rawValue, stats.fps, stats.predictP50, stats.predictP99,
                                stats.resizeP50, BenchmarkRunner.thermal()))
        soakStatus = String(format: "Recording soak log: %d:%02d, thermal %@", seconds / 60, seconds % 60,
                            BenchmarkRunner.thermal())
    }

    private func saveSoakLog() {
        do {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let url = documents.appendingPathComponent("soak_\(BenchmarkRunner.deviceName)_\(stamp).csv")
            try soakLines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
            soakStatus = "Saved \(url.path)"
        } catch {
            soakStatus = "Could not save soak log: \(error.localizedDescription)"
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
