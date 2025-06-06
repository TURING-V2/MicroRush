const std = @import("std");
const stat_calc_lib = @import("../stat_calc/lib.zig");
const StatCalc = stat_calc_lib.StatCalc;
const SymbolMap = @import("../symbol-map.zig").SymbolMap;

pub const SignalEngine = struct {
    allocator: std.mem.Allocator,
    stat_calc: *StatCalc,
    symbol_map: *const SymbolMap,

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) !SignalEngine {
        const device_id = try stat_calc_lib.selectBestCUDADevice();
        var stat_calc = try StatCalc.init(allocator, device_id);
        try stat_calc.getDeviceInfo();
        try stat_calc.warmUp();

        return SignalEngine{
            .allocator = allocator,
            .stat_calc = &stat_calc,
            .symbol_map = symbol_map,
        };
    }

    pub fn deinit(self: *SignalEngine) void {
        self.stat_calc.deinit();
    }

    pub fn run(self: *SignalEngine) !void {
        try self.stat_calc.calculateSymbolMapBatch(self.symbol_map, 6);
    }
};
