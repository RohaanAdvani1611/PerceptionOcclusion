import CoreML

/// Where Core ML plans to run a model's ops on this device, from MLComputePlan.
/// It's a prediction made at load time; measured latency checks it.
nonisolated struct ComputePlanSummary: Sendable {
    var operations = 0
    var opsOn: [String: Int] = [:]        // "ANE", "GPU", "CPU" -> op count
    var costOn: [String: Double] = [:]    // share of estimated cost, 0...1

    func percentOps(_ device: String) -> Double {
        operations == 0 ? 0 : 100 * Double(opsOn[device, default: 0]) / Double(operations)
    }
    func percentCost(_ device: String) -> Double { 100 * costOn[device, default: 0] }

    var text: String {
        ["ANE", "GPU", "CPU"].filter { opsOn[$0, default: 0] > 0 }
            .map { String(format: "%@ %d/%d ops (%.0f%% cost)", $0, opsOn[$0]!, operations, percentCost($0)) }
            .joined(separator: ", ")
    }

    static func load(_ compiledURL: URL, compute: ComputeChoice) async throws -> ComputePlanSummary {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = compute.units
        let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: configuration)
        guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
            return ComputePlanSummary()
        }
        var summary = ComputePlanSummary()
        var totalCost = 0.0
        var cost: [String: Double] = [:]
        for operation in main.block.operations {
            // Constants that hold weights have no device
            guard let usage = plan.deviceUsage(for: operation) else { continue }
            let device = name(of: usage.preferred)
            let weight = plan.estimatedCost(of: operation)?.weight ?? 0
            summary.operations += 1
            summary.opsOn[device, default: 0] += 1
            cost[device, default: 0] += weight
            totalCost += weight
        }
        summary.costOn = cost.mapValues { totalCost > 0 ? $0 / totalCost : 0 }
        return summary
    }

    private static func name(of device: MLComputeDevice) -> String {
        switch device {
        case .neuralEngine: "ANE"
        case .gpu: "GPU"
        case .cpu: "CPU"
        @unknown default: "other"
        }
    }
}
