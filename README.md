## moshi-swift

```bash
xcodebuild -scheme moshi-cli -derivedDataPath ./build
./build/Build/Products/Debug/moshi-cli
```

`LD_RUNPATH_SEARCH_PATHS` has been set in xcode to include the executable path
which is where the `moshi-lib` framework seems to be compiled.
