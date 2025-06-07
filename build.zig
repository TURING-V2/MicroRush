const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const websocket_dep = b.dependency("websocket", .{});

    const exe = b.addExecutable(.{
        .name = "MicroRush",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("websocket", websocket_dep.module("websocket"));

    exe.addIncludePath(b.path("src/cuda/"));
    exe.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    exe.linkLibC();
    exe.linkSystemLibrary("cudart");
    exe.addObjectFile(b.path("kernel.o"));

    exe.addObjectFile(b.path("simd.o"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
