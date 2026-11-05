import SwiftUI

struct ContentView: View {
    @State private var app = AppModel()

    var body: some View {
        VStack(spacing: 12) {
            if let renderer = app.renderer {
                // Camera frames are 4:3; the 256x256 depth is stretched back to that shape
                DepthView(renderer: renderer)
                    .aspectRatio(4.0 / 3.0, contentMode: .fit)
            } else {
                Text("Metal is not available on this device.")
            }

            HStack {
                Picker("Model", selection: $app.variant) {
                    ForEach(ModelVariant.allCases) { Text($0.label).tag($0) }
                }
                Picker("Compute", selection: $app.compute) {
                    ForEach(ComputeChoice.allCases) { Text($0.label).tag($0) }
                }
            }
            .disabled(app.isLoading || app.isBenchmarking)

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow {
                    Text("Camera")
                    Text("\(app.stats.cameraWidth)×\(app.stats.cameraHeight)")
                }
                GridRow {
                    Text("FPS")
                    Text(String(format: "%.0f", app.stats.fps))
                }
                GridRow {
                    Text("Model p50 / p99")
                    Text(String(format: "%.2f / %.2f ms", app.stats.predictP50, app.stats.predictP99))
                }
                GridRow {
                    Text("Resize p50")
                    Text(String(format: "%.2f ms", app.stats.resizeP50))
                }
                GridRow {
                    Text("Load")
                    if app.isLoading {
                        Text("loading…")
                    } else if let info = app.loadInfo {
                        Text(String(format: "%.0f ms, %.1f MB compiled", info.loadMilliseconds, info.compiledMegabytes))
                    }
                }
                GridRow {
                    Text("Planned on")
                    Text(app.planText ?? "…")
                }
            }
            .font(.body.monospacedDigit())

            if let message = app.errorMessage {
                Text(message).foregroundStyle(.red)
            }

            Button("Run benchmark (all 16 combinations, about 5 minutes)") { app.runBenchmark() }
                .disabled(app.isLoading || app.isBenchmarking)
            if let status = app.benchmarkStatus {
                Text(status).font(.caption).textSelection(.enabled)
            }
        }
        .padding()
        .task { await app.start() }
        .onChange(of: app.variant) { app.reload() }
        .onChange(of: app.compute) { app.reload() }
    }
}

#Preview {
    ContentView()
}
