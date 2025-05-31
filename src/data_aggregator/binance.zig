const std = @import("std");
const http = std.http;
const time = std.time;
const json = std.json;

const _ = @import("../errors.zig");
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const Symbol = @import("../types.zig").Symbol;
const WSClient = @import("binance_ws.zig").WSClient;

const REST_ENDPOINTS = [6][]const u8{
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
};

const PING_API = "/api/v3/ping";
const EXCHANGE_INFO_API = "/api/v3/exchangeInfo";
const TIME_API = "/api/v3/time";

pub const Client = struct {
    selected_endpoint: []const u8,
    http_client: http.Client,
    ws_client: WSClient,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Client {
        return Client{
            .selected_endpoint = REST_ENDPOINTS[0],
            .http_client = http.Client{
                .allocator = allocator,
            },
            .ws_client = try WSClient.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http_client.deinit();
        self.ws_client.deinit();
    }

    pub fn connect(self: *Client) !void {
        try self.selectBestEndpoint();
        std.log.info("Connecting to Binance using endpoint: {s}", .{self.selected_endpoint});
    }

    pub fn loadSymbols(self: *Client, sym_map: *SymbolMap) !void {
        std.log.info("Loading symbols from exchange info...", .{});
        var exchange_url_buf: [256]u8 = undefined;
        const exchange_url = try std.fmt.bufPrint(&exchange_url_buf, "{s}{s}", .{ self.selected_endpoint, EXCHANGE_INFO_API });
        const uri = try std.Uri.parse(exchange_url);
        var req = try self.http_client.open(.GET, uri, .{
            .server_header_buffer = try self.allocator.alloc(u8, 8192),
        });
        defer req.deinit();

        try req.send();
        try req.wait();

        if (req.response.status != .ok) {
            return error.ExchangeInfoRequestFailed;
        }
        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024 * 20);
        defer self.allocator.free(body);
        try self.parseAndStoreSymbols(body, sym_map);

        std.log.info("Loaded {} symbols", .{sym_map.count()});
    }

    fn parseAndStoreSymbols(self: *Client, json_data: []const u8, sym_map: *SymbolMap) !void {
        var parsed = json.parseFromSlice(json.Value, self.allocator, json_data, .{}) catch |err| {
            std.log.err("Failed to parse JSON: {}", .{err});
            return err;
        };
        defer parsed.deinit();
        const root = parsed.value;

        const symbols_array = root.object.get("symbols") orelse {
            return error.NoSymbolsFound;
        };
        if (symbols_array != .array) {
            return error.InvalidSymbolsFormat;
        }

        for (symbols_array.array.items) |symbol_value| {
            if (symbol_value != .object) continue;
            const symbol_obj = symbol_value.object;

            // only USDT pairs
            const symbol_name_val = symbol_obj.get("symbol") orelse continue;
            if (symbol_name_val != .string) continue;
            const symbol_str = symbol_name_val.string;
            if (symbol_str.len < 5 or !std.mem.endsWith(u8, symbol_str, "USDT")) continue;
            // only "status": "TRADING"
            const status_val = symbol_obj.get("status") orelse continue;
            if (status_val != .string or !std.mem.eql(u8, status_val.string, "TRADING")) continue;
            // only SPOT Trading tokens
            const permission_sets_val = symbol_obj.get("permissionSets") orelse continue;
            if (permission_sets_val != .array) continue;

            var found_spot = false;
            for (permission_sets_val.array.items) |permission_set_group| {
                if (permission_set_group != .array) continue;
                for (permission_set_group.array.items) |perm_val| {
                    if (perm_val != .string) continue;
                    if (std.mem.eql(u8, perm_val.string, "SPOT")) {
                        found_spot = true;
                        break;
                    }
                }
                if (found_spot) break;
            }
            if (!found_spot) continue;

            const owned_symbol = try self.allocator.dupe(u8, symbol_str);
            const empty_symbol = Symbol.init();
            try sym_map.put(owned_symbol, empty_symbol);
        }
    }

    fn selectBestEndpoint(self: *Client) !void {
        var ping_ms = [_]u64{0} ** REST_ENDPOINTS.len;
        std.log.info("Testing ping for {} Binance REST_ENDPOINTS...", .{REST_ENDPOINTS.len});
        for (REST_ENDPOINTS, 0..) |endpoint, i| {
            const ping_result = self.pingEndpoint(endpoint) catch |err| {
                std.log.warn("Failed to ping {s}: {}", .{ endpoint, err });
                continue;
            };

            ping_ms[i] = ping_result;
            std.log.info("Endpoint {s}: {}ms", .{ endpoint, ping_result });
        }
        var best_endpoint: ?[]const u8 = null;
        var lowest_ping: u64 = std.math.maxInt(u64);
        for (REST_ENDPOINTS, 0..) |endpoint, i| {
            if (ping_ms[i] < lowest_ping) {
                lowest_ping = ping_ms[i];
                best_endpoint = endpoint;
            }
        }
        if (best_endpoint) |endpoint| {
            self.selected_endpoint = endpoint;
            std.log.info("Selected best endpoint: {s} ({}ms)", .{ endpoint, lowest_ping });
        } else {
            std.log.warn("No REST_ENDPOINTS responded, using default: {s}", .{REST_ENDPOINTS[0]});
            self.selected_endpoint = REST_ENDPOINTS[0];
        }
    }

        const     fn pingEndpoint(self: *Client, endpoint: []const u8) !u64 {
start_time = time.nanoTimestamp();
        var ping_url_buf: [256]u8 = undefined;
        const ping_url = try std.fmt.bufPrint(&ping_url_buf, "{s}{s}", .{ endpoint, PING_API });
        const uri = try std.Uri.parse(ping_url);

        var req = try self.http_client.open(.GET, uri, .{
            .server_header_buffer = try self.allocator.alloc(u8, 4096),
        });
        defer req.deinit();

        try req.send();
        try req.wait();

        const end_time = time.nanoTimestamp();
        const ping_ms = @as(u64, @intCast(@divTrunc(end_time - start_time, time.ns_per_ms)));
        if (req.response.status != .ok) {
            return error.BadResponse;
        }
        return ping_ms;
    }
};
