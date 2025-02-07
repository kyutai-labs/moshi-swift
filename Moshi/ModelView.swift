// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//
//  SwiftUIView.swift
//  Moshi
//
//  Created by Sebastien Mazare on 06/12/2024.
//

import Hub
import MLX
import MLXNN
import MLXRandom
import Metal
import MoshiLib
import SwiftUI
import Synchronization
import AVFoundation

struct ModelView: View {
    @Binding var model: Evaluator
    let modelType: ModelSelect
    @Environment(DeviceStat.self) private var deviceStat
    @Binding var displayStats: Bool
    @State var sendToSpeaker = false

    var body: some View {
        VStack(spacing: 16) {
            ControlBar(
                isRunning: model.running,
                sendToSpeaker: $sendToSpeaker,
                onStart: generate,
                onStop: stopGenerate,
                onSpeakerChange: { newValue in
                    if newValue {
                        setDefaultToSpeaker()
                    } else {
                        setDefaultToStd()
                    }
                }
            )
            
            if model.running || !model.modelInfo.isEmpty {
                StatusSection(model: model)
            }
            
            OutputSection(output: model.output)
            
            if model.statsSummary.encode.cnt > 0 {
                StatsView(summary: model.statsSummary)
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 12).fill(.secondary.opacity(0.1)))
            }
        }
        .padding()
        .navigationTitle("Moshi: \(modelType.rawValue)")
        .toolbar {
            ToolbarItem {
                Button(action: { displayStats.toggle() }) {
                    Label("Stats", systemImage: "chart.bar.fill")
                }
                .popover(isPresented: $displayStats) {
                    DeviceStatsView(deviceStat: deviceStat)
                }
            }
        }
    }

    private func generate() {
        Task {
            await model.generate(self.modelType)
        }
    }

    private func stopGenerate() {
        Task {
            await model.stopGenerate()
        }
    }
}

struct ControlBar: View {
    let isRunning: Bool
    @Binding var sendToSpeaker: Bool
    let onStart: () -> Void
    let onStop: () -> Void
    let onSpeakerChange: (Bool) -> Void
    
    var body: some View {
        HStack {
            Button(action: isRunning ? onStop : onStart) {
                Label(isRunning ? "Stop" : "Start", 
                      systemImage: isRunning ? "stop.circle.fill" : "play.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.borderedProminent)
            
            Spacer()
            
            HStack {
                Image(systemName: "speaker.wave.2")
                Toggle("Use External Speaker", isOn: $sendToSpeaker)
                    .toggleStyle(.switch)
            }
            .onChange(of: sendToSpeaker) { newValue in
                onSpeakerChange(newValue)
            }
        }
        .padding()
    }
}

struct StatusSection: View {
    let model: Evaluator
    
    var body: some View {
        VStack(spacing: 8) {
            if let progress = model.progress {
                ProgressView(progress)
                    .transition(.slide)
            }
            
            if model.running {
                HStack(spacing: 12) {
                    Label("\(Int(model.totalDuration))s", systemImage: "clock")
                    
                    Image(systemName: "microphone.circle.fill")
                        .foregroundStyle(.blue)
                        .font(.title2)
                        .symbolEffect(.bounce, options: .repeating)
                    
                    VStack(alignment: .leading) {
                        Text("Buffer")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Gauge(value: model.bufferedDuration * 1000.0, in: 0...500) {
                            EmptyView()
                        } currentValueLabel: {
                            Text("\(Int(model.bufferedDuration * 1000.0))ms")
                        }
                        .tint(.blue)
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 10).fill(.blue.opacity(0.1)))
            }
        }
    }
}

struct OutputSection: View {
    let output: String
    
    var body: some View {
        ScrollView(.vertical) {
            ScrollViewReader { proxy in
                Text(output)
                    .textSelection(.enabled)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .onChange(of: output) { _, _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                
                Color.clear
                    .frame(height: 1)
                    .id("bottom")
            }
        }
        .background(RoundedRectangle(cornerRadius: 12).fill(.secondary.opacity(0.1)))
    }
}

// Move these to separate files
struct DeviceStatsView: View {
    let deviceStat: DeviceStat
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Device Statistics")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                MemoryStatRow(
                    label: "Active Memory",
                    value: deviceStat.gpuUsage.activeMemory,
                    total: GPU.memoryLimit
                )
                MemoryStatRow(
                    label: "Cache Memory",
                    value: deviceStat.gpuUsage.cacheMemory,
                    total: GPU.cacheLimit
                )
                MemoryStatRow(
                    label: "Peak Memory",
                    value: deviceStat.gpuUsage.peakMemory
                )
            }
        }
        .padding()
        .frame(minWidth: 300)
    }
}

struct MemoryStatRow: View {
    let label: String
    let value: Int
    var total: Int?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .foregroundStyle(.secondary)
            if let total = total {
                Gauge(value: Double(value), in: 0...Double(total)) {
                    Text(value.formatted(.byteCount(style: .memory)))
                }
                .tint(.blue)
            } else {
                Text(value.formatted(.byteCount(style: .memory)))
            }
        }
    }
}

struct StatsView: View {
    let summary: StatsSummary
    
    var body: some View {
        Grid(alignment: .leading) {
            GridRow {
                Text("Step")
                Text("Avg (ms)")
                Text("Min (ms)")
                Text("Max (ms)")
                Text("Count")
            }
            .bold()
            
            Divider()
            
            StatRow(label: "Encode", stats: summary.encode)
            StatRow(label: "Main", stats: summary.step)
            StatRow(label: "Depformer", stats: summary.depformer)
            StatRow(label: "Decode", stats: summary.decode)
        }
        .font(.system(.body, design: .monospaced))
    }
}

struct StatRow: View {
    let label: String
    let stats: StatsSummary.Stats
    
    var body: some View {
        GridRow {
            Text(label)
            Text((1000 * stats.sum / Float(stats.cnt)).rounded(), format: .number)
            Text((1000 * stats.min).rounded(), format: .number)
            Text((1000 * stats.max).rounded(), format: .number)
            Text(stats.cnt, format: .number)
        }
    }
}
