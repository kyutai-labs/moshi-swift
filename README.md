## moshi-swift

This repo contains an experimental implementation of
[mimi](https://huggingface.co/kyutai/mimi) and
[moshi](https://github.com/kyutai-labs/moshi) using [MLX
Swift](https://github.com/ml-explore/mlx-swift).

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


