fmt:
	zig fmt .

build:
	clear && zig build

build-fast:
	clear && zig build -Doptimize=ReleaseFast -Dtarget=native -Dcpu=native

build-safe:
	clear && zig build -Doptimize=ReleaseSafe -Dtarget=native -Dcpu=native

run:
	./zig-out/bin/MicroRush

run-metrics:
	./zig-out/bin/MicroRush --metrics

clean:
	rm -rf zig-out
	rm -rf .zig-cache

.PHONY: fmt build-fast run clean
