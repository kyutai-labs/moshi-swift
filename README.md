## moshi-swift

```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
wget https://huggingface.co/kyutai/moshika-candle-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors
xcodebuild -scheme moshi-cli -derivedDataPath ./build
./build/Build/Products/Debug/MoshiCLI
```

`LD_RUNPATH_SEARCH_PATHS` has been set in xcode to include the executable path
which is where the `moshi-lib` framework seems to be compiled.

When running on the command line via ssh, this may require unlocking the keychain with:
```
security unlock-keychain
```

Added to `OTHER_SWIFT_FLAGS` `-no-verify-emitter-module-interface`,
as per [github issue](https://github.com/swiftlang/swift/issues/64669).
