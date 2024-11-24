// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX
import MLXNN
import MoshiLib

let homeDirectory = NSHomeDirectory()

func runTransformer() throws {
    let weights = try loadArrays(
        url: URL(fileURLWithPath: homeDirectory + "/tmp/model.safetensors"))
    print(weights.keys)
    let parameters = ModuleParameters.unflattened(weights)
    let cfg = LmConfig.moshi_v0_1()
    let model = LM(cfg)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    print(model)
    let token_ids = MLXArray([42], [1, 1])
    let out = model(token_ids)
    print(out.shape, out.dtype, out.ndim)
}

func runMimi() throws {
    let cfg = MimiConfig.v0_1()
    let model = Mimi(cfg)

    let origWeights = try loadArrays(
        url: URL(
            fileURLWithPath: homeDirectory + "/tmp/tokenizer-e351c8d8-checkpoint125.safetensors"))
    var weights: [String: MLXArray] = [:]
    for (var key, var weight) in origWeights {
        // Mutating the keys while iterating over the map seems pretty dodgy, not sure what the idiomatic
        // way to do this is in swift. Hopefully this is copy on write and it's all good :)
        if key.hasPrefix("encoder.model") {
            key.replace("encoder.model.", with: "encoder.")
        }
        if key.hasPrefix("decoder.model") {
            key.replace("decoder.model.", with: "decoder.")
        }
        if key.hasSuffix(".in_proj_weight") {
            key.replace(".in_proj_weight", with: ".in_proj.weight")
        }
        if key.hasSuffix(".linear1.weight") {
            key.replace(".linear1.weight", with: ".gating.linear1.weight")
        }
        if key.hasSuffix(".linear2.weight") {
            key.replace(".linear2.weight", with: ".gating.linear2.weight")
        }
        // Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
        for (layerIdx, decoderIdx) in [2, 5, 8, 11].enumerated() {
            key.replace("decoder.\(decoderIdx).", with: "decoder.layers.\(layerIdx).upsample.")
            key.replace(
                "decoder.\(decoderIdx + 1).", with: "decoder.layers.\(layerIdx).residuals.0.")
        }
        for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
            key.replace("encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
            key.replace(
                "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
        }
        key.replace("decoder.0.", with: "decoder.init_conv1d.")
        key.replace("decoder.14.", with: "decoder.final_conv1d.")
        key.replace("encoder.0.", with: "encoder.init_conv1d.")
        key.replace("encoder.14.", with: "encoder.final_conv1d.")
        key.replace(".block.1.", with: ".block.0.")
        key.replace(".block.3.", with: ".block.1.")
        // PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
        if key.hasSuffix(".conv.weight") || key.hasSuffix(".output_proj.weight")
            || key.hasSuffix(".input_proj.weight")
        {
            weight = weight.swappedAxes(-1, -2)
        }
        // PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
        if key.hasSuffix(".convtr.weight") {
            weight = weight.transposed(axes: [1, 2, 0])
        }

        print(key, weight.shape)
        weights[key] = weight
    }
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    let input = MLXArray.zeros([1, 1, 24000], dtype: .float32)
    let out = model.encode(input)
    print("quantized", out.shape)
    print(out)
    let pcm = model.decode(out)
    print("pcm", pcm.shape)
}

try runMimi()
