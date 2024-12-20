import Foundation
import MLX

@Observable
final class DeviceStat: @unchecked Sendable {

    @MainActor
    var gpuUsage = GPU.snapshot()
    @MainActor
    var thermalState: ProcessInfo.ThermalState = .nominal

    private let initialGPUSnapshot = GPU.snapshot()
    private var timer: Timer?

    init() {
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateGPUUsages()
        }
    }

    deinit {
        timer?.invalidate()
    }

    private func updateGPUUsages() {
        let gpuSnapshotDelta = initialGPUSnapshot.delta(GPU.snapshot())
        let thermalState = ProcessInfo.processInfo.thermalState
        DispatchQueue.main.async { [weak self] in
            self?.gpuUsage = gpuSnapshotDelta
            self?.thermalState = thermalState
        }
    }

}
