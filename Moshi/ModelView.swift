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

struct ModelView: View {
    @Binding var model: Evaluator
    let modelType: ModelSelect
    @Environment(DeviceStat.self) private var deviceStat
    @Binding var displayStats: Bool

    var body: some View {
        let gridRow = { (name: String, ss: StatsSummary.Stats) -> GridRow in
            GridRow {
                Text(name)
                    .font(.system(.body, design: .monospaced))
                Text((1000 * ss.sum / Float(ss.cnt)).rounded(), format: .number)
                    .font(.system(.body, design: .monospaced))
                Text((1000 * ss.min).rounded(), format: .number)
                    .font(.system(.body, design: .monospaced))
                Text((1000 * ss.max).rounded(), format: .number)
                    .font(.system(.body, design: .monospaced))
                Text(ss.cnt, format: .number)
                    .font(.system(.body, design: .monospaced))
            }
        }
        let summaryTable: Grid =
            Grid {
                GridRow {
                    Text("step")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                    Text("avg")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                    Text("min")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                    Text("max")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                    Text("cnt")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                }
                gridRow("Enc", model.statsSummary.encode)
                gridRow("Main", model.statsSummary.step)
                gridRow("DepF", model.statsSummary.depformer)
                gridRow("Dec", model.statsSummary.decode)
            }
        let thermalState =
            switch deviceStat.thermalState {
            case .critical: "critical"
            case .fair: "fair"
            case .nominal: "nominal"
            case .serious: "serious"
            case let other: "\(other)"
            }
        return VStack {
            VStack {

                Text(model.modelInfo)
                    .font(.system(size: 22.0, weight: .semibold))
                    .padding()

                if let progress = model.progress {
                    ProgressView(progress)
                        .padding()
                        .transition(.slide)
                }
                if model.running {
                    HStack {
                        Image(systemName: "microphone.circle")
                            .font(.system(size: 40))
                            .symbolEffect(.breathe)
                        Gauge(value: model.bufferedDuration * 1000.0, in: 0...500) {
                        } currentValueLabel: {
                            Text("\(Int(model.bufferedDuration * 1000.0))")
                        }
                        .gaugeStyle(.accessoryCircular)
                    }
                }
                HStack {
                    Spacer()
                    Button(
                        model.running ? "Stop" : "Run",
                        action: withAnimation { model.running ? stopGenerate : generate }
                    )
                    .buttonStyle(BorderedButtonStyle())
                    .padding()
                    Spacer()
                }
                if let (traceURL, codesURL) = model.urls {
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
            .background(RoundedRectangle(cornerRadius: 15.0).fill(.blue.opacity(0.1)))
            .padding()

            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        Text(model.output)
                            .textSelection(.enabled)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .onChange(of: model.output) { _, _ in
                        sp.scrollTo("bottom")
                    }
                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(RoundedRectangle(cornerRadius: 15.0).fill(.blue.opacity(0.1)))
            .padding()

            summaryTable
                .padding()
            Text("thermal: \(thermalState)")
        }
        .toolbar {
            ToolbarItem {
                Button(
                    action: { displayStats.toggle() },
                    label: {
                        Label(
                            "Stats",
                            systemImage: "info.circle.fill"
                        )
                    }
                )
                .popover(isPresented: $displayStats) {
                    VStack {
                        HStack {
                            Text(
                                "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))"
                            )
                            .bold()
                            Spacer()
                        }
                        Text(
                            """
                            Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                            Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                            Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                            """
                        )
                        .font(.callout)
                        .italic()
                    }
                    .frame(minWidth: 200.0)
                    .padding()
                }
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
            }
        }
        .navigationTitle("Moshi : \(modelType.rawValue)")
        #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
        #endif
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
