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
    @State var sendToSpeaker = false
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 16) {
            if model.running || !model.modelInfo.isEmpty {
                StatusSection(model: model)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
            
            OutputSection(output: model.output)
            
            if deviceStat.gpuUsage.activeMemory > 0 || model.statsSummary.step.cnt > 0 {
                CombinedStatsView(summary: model.statsSummary, deviceStat: deviceStat)
                    .frame(height: 250)
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 12).fill(.secondary.opacity(0.1)))
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            // Bottom controls
            ZStack {
                // Centered Start/Stop button
                Button(action: model.running ? stopGenerate : generate) {
                    Label(model.running ? "Stop" : "Start", 
                          systemImage: model.running ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title2)
                }
                .buttonStyle(.borderedProminent)
                
                // Right-aligned settings button
                HStack {
                    Spacer()
                    Button(action: { showSettings.toggle() }) {
                        Image(systemName: "gear")
                            .font(.title2)
                    }
                    .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                        Toggle(isOn: $sendToSpeaker) {
                            Label("Use External Speaker", systemImage: "speaker.wave.2")
                        }
                        .padding()
                        .onChange(of: sendToSpeaker) { (_, newValue) in
                            if newValue {
                                setDefaultToSpeaker()
                            } else {
                                setDefaultToStd()
                            }
                        }
                        .presentationCompactAdaptation(.popover)
                    }
                }
            }
            .padding()
        }
        .padding()
        .animation(.smooth, value: model.running)
        .animation(.smooth, value: model.statsSummary.encode.cnt)
        .navigationTitle("Moshi: \(modelType.rawValue)")
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

struct DeviceStatsView: View {
    let deviceStat: DeviceStat
    
    var body: some View {
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

struct CombinedStatsView: View {
    let summary: StatsSummary
    let deviceStat: DeviceStat
    @State private var currentPage = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with page indicator
            HStack {
                Spacer()
                HStack(spacing: 16) {
                    ForEach(0..<2) { index in
                        Button(action: { withAnimation { currentPage = index } }) {
                            VStack(spacing: 4) {
                                Text(index == 0 ? "Model Stats" : "Device Stats")
                                    .font(.subheadline)
                                    .foregroundStyle(currentPage == index ? .primary : .secondary)
                                
                                Rectangle()
                                    .fill(currentPage == index ? .blue : .clear)
                                    .frame(height: 2)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                Spacer()
            }
            .padding(.bottom, 8)
            
            TabView(selection: $currentPage) {
                StatsView(summary: summary)
                    .padding(.vertical)
                    .frame(height: 250)
                    .tag(0)
                
                DeviceStatsView(deviceStat: deviceStat)
                    .padding(.vertical)
                    .tag(1)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
        }
    }
}
