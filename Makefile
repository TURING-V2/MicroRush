fmt:
	zig fmt .

build:
	clear && zig build

build-fast:
	zig build -Doptimize=ReleaseFast -Dtarget=native -Dcpu=znver2

run:
	./zig-out/bin/MicroRush

clean: 
	rm -rf zig-out
	rm -rf .zig-cache

.PHONY: fmt build run 

