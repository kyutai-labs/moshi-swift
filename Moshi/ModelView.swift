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
                Text((1000 * ss.sum / Float(ss.cnt)).rounded(), format: .number)
                Text((1000 * ss.min).rounded(), format: .number)
                Text((1000 * ss.max).rounded(), format: .number)
                Text(ss.cnt, format: .number)
            }
        }
        let summaryTable: Grid =
            Grid {
                GridRow {
                    Text("step")
                    Text("avg (ms)")
                    Text("min (ms)")
                    Text("max (ms)")
                    Text("cnt")
                }
                gridRow("MimiEnc", model.statsSummary.encode)
                gridRow("MainLM", model.statsSummary.step)
                gridRow("DepFormer", model.statsSummary.depformer)
                gridRow("MimiDec", model.statsSummary.decode)
            }
        return VStack {
            VStack {

                Text(model.modelInfo)
                    .font(.system(size: 22.0, weight: .semibold))
                    .padding()

                if model.progress != nil {
                    ProgressView(model.progress!)
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
                if let traceURL = model.traceURL {
                    ShareLink(item: traceURL) {
                        Label("Share Trace", systemImage: "square.and.arrow.up")
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
            .background(RoundedRectangle(cornerRadius: 15.0).fill(.blue.opacity(0.1)))
            .padding()
            summaryTable
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
