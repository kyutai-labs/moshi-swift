// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import SwiftUI

func requestMicrophoneAccess() {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return
    case .notDetermined:
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            print("granted", granted)
        }
    case .denied:  // The user has previously denied access.
        return
    case .restricted:  // The user can't grant access due to restrictions.
        return
    case _:
        return
    }
}

@main
struct moshiApp: App {
    @Environment(\.scenePhase) var scenePhase

    init() {
        requestMicrophoneAccess()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
#if os(iOS)
        .onChange(of: scenePhase) { (phase) in
            switch phase {
            case .active:
                // In iOS 13+, idle timer needs to be set in scene to override default
                UIApplication.shared.isIdleTimerDisabled = true
            case .inactive: break
            case .background: break
            @unknown default: print("ScenePhase: unexpected state")
            }
        }
#endif
    }
}
