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
            Group {
                if deviceStat.gpuUsage.activeMemory > 0 || model.statsSummary.step.cnt > 0 {
                    CombinedStatsView(
                        summary: model.statsSummary,
                        deviceStat: deviceStat,
                        modelInfo: model.modelInfo,
                        modelName: model.modelName,
                        urls: model.urls
                    )
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 12).fill(.secondary.opacity(0.1)))
                }
            }
            .transition(.push(from: .top))
            .animation(.easeInOut(duration: 0.2), value: deviceStat.gpuUsage.activeMemory > 0 || model.statsSummary.step.cnt > 0)

            Group {
                if !model.running && model.output.isEmpty {
                    ModelInfoView(modelType: modelType)
                } else {
                    OutputSection(output: model.output)
                }
            }
            .transition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: !model.running && model.output.isEmpty)

            Group {
                if model.running || !model.output.isEmpty {
                    StatusSection(model: model)
                }
            }
            .transition(.push(from: .bottom))
            .animation(.easeInOut(duration: 0.2), value: model.running || !model.output.isEmpty)

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
        .navigationTitle("Moshi: \(modelType.name)")
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
                let tint = model.totalDuration > 0 ? Color.blue : Color.red
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
                .background(RoundedRectangle(cornerRadius: 10).fill(tint.opacity(0.1)))
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
            HStack {
                MemoryStatRow(
                    label: "Peak Memory",
                    value: deviceStat.gpuUsage.peakMemory
                )
                Spacer()
                VStack(alignment: .leading) {
                    Text("Thermal State")
                        .foregroundStyle(.secondary)
                    Text(deviceStat.thermalState.rawValue)
                }
            }

        }
    }
}

struct DebugView: View {
    let modelInfo: String
    let modelName: String
    let urls: (URL, URL)?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Last Info").bold()
            Text(modelInfo)
            Text("Model Name").bold()
            Text(modelName)
            if let (traceURL, codesURL) = urls {
                HStack {
                    ShareLink(item: traceURL) {
                        Label("Trace", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(BorderedButtonStyle())
                        ShareLink(item: codesURL) {
                            Label("Codes", systemImage: "square.and.arrow.up")
                        }
                    .buttonStyle(BorderedButtonStyle())
                }
                .padding()
            }
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
    let modelInfo: String
    let modelName: String
    let urls: (URL, URL)?
    @State private var currentPage = 0
    @State private var isExpanded = true

    var body: some View {
        VStack(spacing: 0) {
            // Header with page indicator and collapse button
            ZStack {
                // Left-aligned content
                HStack {
                    if isExpanded {
                        HStack(spacing: 16) {
                            ForEach(0..<3) { index in
                                Button(action: { withAnimation { currentPage = index } }) {
                                    let text = switch index {
                                    case 0: "Model"
                                    case 1: "Device"
                                    case 2: "Details"
                                    case _: "unk"
                                    }
                                    VStack(spacing: 4) {
                                        Text(text)
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
                        .frame(maxWidth: .infinity)
                    } else {
                        Text("Details")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                // Right-aligned button (always in the same position)
                HStack {
                    Spacer()
                    Button(action: {
                        isExpanded.toggle()
                    }) {
                        Image(systemName: isExpanded ? "chevron.up.circle.fill" : "chevron.down.circle.fill")
                            .foregroundStyle(.secondary)
                            .font(.title3)
                    }
                }
            }
            .padding(.bottom, isExpanded ? 8 : 0)
            // Add tap gesture only when collapsed
            .contentShape(Rectangle()) // Make entire area tappable
            .onTapGesture {
                if !isExpanded {
                    withAnimation {
                        isExpanded.toggle()
                    }
                }
            }

            if isExpanded {
                TabView(selection: $currentPage) {
                    StatsView(summary: summary)
                        .padding(.vertical)
                        .frame(height: 250)
                        .tag(0)
                    DeviceStatsView(deviceStat: deviceStat)
                        .padding(.vertical)
                        .tag(1)
                    DebugView(modelInfo: modelInfo, modelName: modelName, urls: urls)
                        .padding(.vertical)
                        .tag(2)
                }
                #if os(iOS)
                .tabViewStyle(.page(indexDisplayMode: .never))
                #endif
            }
        }
        .frame(height: isExpanded ? 250 : 44)
        .animation(.easeInOut(duration: 0.2), value: isExpanded)
    }
}

struct ModelInfoView: View {
    let modelType: ModelSelect

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 64))
                .foregroundStyle(.blue)

            VStack(spacing: 12) {
                Text(modelType.name)
                    .font(.title)
                    .bold()

                Text(modelType.description)
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 12).fill(.secondary.opacity(0.1)))
    }
}
