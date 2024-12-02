## moshi-swift

[Moshi][moshi] is a speech-text foundation model and **full-duplex** spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec. Mimi processes 24 kHz audio, down to a 12.5 Hz representation
with a bandwidth of 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size).


This repo contains implementations of these models using [MLX
Swift](https://github.com/ml-explore/mlx-swift):
- Fully streaming implementation of the mimi codec.
- Support for all moshi variants.




Compile and test using the command line:
```bash
xcodebuild -scheme moshi-cli -derivedDataPath ./build
./build/Build/Products/Release/MoshiCLI moshi-7b
```

The checkpoints are automatically downloaded from the huggingface hub. So you
may have to wait for a bit when running the model for the first time.

### Possible workarounds for common issues
`LD_RUNPATH_SEARCH_PATHS` has been set in xcode to include the executable path
which is where the `moshi-lib` framework seems to be compiled.

When running on the command line via ssh, this may require unlocking the keychain with:
```
security unlock-keychain
```

Added to `OTHER_SWIFT_FLAGS` `-no-verify-emitter-module-interface`,
as per [github issue](https://github.com/swiftlang/swift/issues/64669).

## License

The present code is provided under the MIT license.
The weights for the models are released under the CC-BY 4.0 license.

## Citation

If you use either Mimi or Moshi, please cite the following paper,

```
@techreport{kyutai2024moshi,
      title={Moshi: a speech-text foundation model for real-time dialogue},
      author={Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and
      Am\'elie Royer and Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
      year={2024},
      eprint={2410.00037},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.00037},
}
```

[moshi]: https://arxiv.org/abs/2410.00037
