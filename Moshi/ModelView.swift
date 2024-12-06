//
//  SwiftUIView.swift
//  Moshi
//
//  Created by Sebastien Mazare on 06/12/2024.
//

import SwiftUI
import Hub
import MLX
import MLXNN
import MLXRandom
import Metal
import MoshiLib
import Synchronization

struct ModelView: View {
    @Binding var model: Evaluator
    let modelType: ModelSelect
    @Environment(DeviceStat.self) private var deviceStat
    @Binding var displayStats: Bool
    
    var body: some View {
        VStack {
            VStack {

                Text(model.modelInfo)
                    .font(.system(size: 22.0, weight: .semibold))
                    .padding()

                if model.progress != nil {
                    ProgressView(model.progress!)
                        .padding()
                        .transition(.slide)
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
            }
            .background(RoundedRectangle(cornerRadius: 15.0).fill(.blue.opacity(0.1)))
            .padding()

            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        Text(model.output)
                            .textSelection(.enabled)
                    }
                    .onChange(of: model.output) { _, _ in
                        sp.scrollTo("bottom")
                    }
                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
            .padding()
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
                    .frame(minWidth:200.0)
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
