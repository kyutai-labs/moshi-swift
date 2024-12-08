.PHONY: format run-1b run-asr

format:
	swift-format format --in-place --recursive .

run-1b:
	xcodebuild -scheme moshi-cli -derivedDataPath ./build
	./build/Build/Products/Release/MoshiCLI moshi-1b-file

run-asr:
	xcodebuild -scheme moshi-cli -derivedDataPath ./build
	./build/Build/Products/Release/MoshiCLI asr-file
