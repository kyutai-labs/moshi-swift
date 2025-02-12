import Foundation
import MLX

enum ThermalState: String {
    case nominal = "Nominal"
    case fair = "Fair"
    case serious = "Serious"
    case critical = "Critical"
    case unknown = "Unknown"
}

@Observable
final class DeviceStat: @unchecked Sendable {

    @MainActor
    var gpuUsage = GPU.snapshot()
    @MainActor
    var thermalState: ThermalState = .nominal

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
            switch thermalState {
                case .nominal:
                    self?.thermalState = .nominal
                case .fair:
                    self?.thermalState = .fair
                case .serious:
                    self?.thermalState = .serious
                case .critical:
                    self?.thermalState = .critical
                default:
                    self?.thermalState = .unknown
            }
        }
    }

}
