## moshi-swift

```bash
xcodebuild -scheme moshi-cli -derivedDataPath ./build
./build/Build/Products/Debug/MoshiCLI
```

`LD_RUNPATH_SEARCH_PATHS` has been set in xcode to include the executable path
which is where the `moshi-lib` framework seems to be compiled.

When running on the command line via ssh, this may require unlocking the keychain with:
```
security unlock-keychain
```
