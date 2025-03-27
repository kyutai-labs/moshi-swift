.PHONY: format run-1b run-asr build

format:
	swift-format format --in-place --recursive .

run-1b: build
	./build/Build/Products/Release/MoshiCLI run

run-asr: build
	./build/Build/Products/Release/MoshiCLI run-asr

run-mimi: build
	./build/Build/Products/Release/MoshiCLI run-mimi

run-helium: build
	./build/Build/Products/Release/MoshiCLI run-helium

run-qwen: build
	./build/Build/Products/Release/MoshiCLI run-qwen

build:
	xcodebuild -scheme moshi-cli -derivedDataPath ./build
