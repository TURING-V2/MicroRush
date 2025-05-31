const std = @import("std");
const json = std.json;
const websocket = @import("websocket");
const http = std.http;

const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const OHLC = @import("../types.zig").OHLC;
const Symbol = @import("../types.zig").Symbol;
const OrderBook = @import("../types.zig").OrderBook;
const DepthError = @import("../errors.zig").DepthError;

const DEPTH_API = "https://api.binance.com/api/v3/depth?symbol={s}&limit=10";

const DepthEvent = struct {
    // event type
    e: []const u8,
    // event time
    E: i64,
    // symbol
    s: []const u8,
    // first update ID in event
    U: i64,
    // final update ID in event
    u: i64,
    // bids [price, quantity]
    b: [][]const []const u8,
    // asks [price, quantity]
    a: [][]const []const u8,
};

const DepthSnapshot = struct {
    lastUpdateId: i64,
    bids: [][]const []const u8,
    asks: [][]const []const u8,
    // add fields to track allocated memory for cleanup
    bids_data: [][]const u8,
    asks_data: [][]const u8,
};

const BufferedDepthEvent = struct {
    event: DepthEvent,
    data: []u8, // keep original JSON data for memory management
};

pub const DepthHandler = struct {
    symbol_map: *SymbolMap,
    allocator: std.mem.Allocator,
    http_client: *http.Client,
    depth_event_buffers: std.StringHashMap(std.ArrayList(BufferedDepthEvent)),
    orderbook_initialized: std.StringHashMap(bool),
    last_update_ids: std.StringHashMap(i64),
    mutex: std.Thread.Mutex = .{},
    message_count: u64,
    last_reset_time: i64,

    pub fn init(symbol_map: *SymbolMap, allocator: std.mem.Allocator, http_client: *http.Client) !DepthHandler {
        return DepthHandler{
            .symbol_map = symbol_map,
            .allocator = allocator,
            .http_client = http_client,
            .depth_event_buffers = std.StringHashMap(std.ArrayList(BufferedDepthEvent)).init(allocator),
            .orderbook_initialized = std.StringHashMap(bool).init(allocator),
            .last_update_ids = std.StringHashMap(i64).init(allocator),
            .message_count = 0,
            .last_reset_time = std.time.milliTimestamp(),
        };
    }

    pub fn deinit(self: *DepthHandler) void {
        var buffer_it = self.depth_event_buffers.iterator();
        while (buffer_it.next()) |entry| {
            for (entry.value_ptr.items) |buffered_event| {
                self.allocator.free(buffered_event.data);
            }
            entry.value_ptr.deinit();
        }
        self.depth_event_buffers.deinit();
        self.orderbook_initialized.deinit();
        self.last_update_ids.deinit();
    }

    pub fn serverMessage(self: *DepthHandler, data: []u8, message_type: websocket.MessageType) !void {
        if (message_type != .text) return;

        self.mutex.lock();
        self.message_count += 1;
        self.mutex.unlock();

        try self.printMessagesPerSecond();

        const json_str = data;
        const parsed = json.parseFromSlice(json.Value, self.allocator, json_str, .{}) catch |err| {
            std.log.err("Failed to parse depth JSON: {}", .{err});
            return;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return;

        const symbol_val = root.object.get("s") orelse return;
        if (symbol_val != .string) return;
        const symbol = try self.allocator.dupe(u8, symbol_val.string);
        defer self.allocator.free(symbol);

        try self.handleDepthUpdate(root, symbol, data);
    }

    fn printMessagesPerSecond(self: *DepthHandler) !void {
        const current_time = std.time.milliTimestamp();
        const elapsed_ms = current_time - self.last_reset_time;

        // Print every ~1000ms (1 second)
        if (elapsed_ms >= 1000) {
            self.mutex.lock();
            const messages = self.message_count;
            self.message_count = 0; // Reset counter
            self.last_reset_time = current_time;
            self.mutex.unlock();

            const messages_per_sec = @as(f64, @floatFromInt(messages)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
            std.debug.print("DepthHandler messages per second: {d:.2}\n", .{messages_per_sec});
        }
    }

    fn handleDepthUpdate(self: *DepthHandler, root: json.Value, symbol: []const u8, original_data: []u8) DepthError!void {
        const U_val = root.object.get("U") orelse return;
        const u_val = root.object.get("u") orelse return;
        if (U_val != .integer or u_val != .integer) return;

        const first_update_id = U_val.integer;
        const last_update_id = u_val.integer;

        // Check if order book is initialized for this symbol
        const is_initialized = self.orderbook_initialized.get(symbol) orelse false;

        if (!is_initialized) {
            // Buffer the event and initialize order book
            try self.bufferDepthEvent(symbol, original_data);
            try self.initializeOrderBook(symbol);
        } else {
            // Apply the update directly
            try self.applyDepthUpdate(symbol, root, first_update_id, last_update_id);
        }
    }

    fn bufferDepthEvent(self: *DepthHandler, symbol: []const u8, data: []u8) !void {
        const result = try self.depth_event_buffers.getOrPut(symbol);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(BufferedDepthEvent).init(self.allocator);
        }

        const data_copy = try self.allocator.dupe(u8, data);

        const buffered_event = BufferedDepthEvent{
            .event = undefined, // Will be parsed when needed
            .data = data_copy,
        };

        try result.value_ptr.append(buffered_event);
    }

    fn initializeOrderBook(self: *DepthHandler, symbol: []const u8) DepthError!void {
        // Step 3: Get depth snapshot
        const snapshot = self.getDepthSnapshot(symbol) catch |err| {
            std.log.err("Failed to get depth snapshot for {s}: {}", .{ symbol, err });
            return DepthError.DepthRequestFailed;
        };
        defer self.freeDepthSnapshot(snapshot);

        // Step 4 & 5: Process buffered events
        if (self.depth_event_buffers.getPtr(symbol)) |buffer| {
            // Find first valid event
            var valid_events = std.ArrayList(usize).init(self.allocator);
            defer valid_events.deinit();

            for (buffer.items, 0..) |buffered_event, i| {
                const parsed = json.parseFromSlice(json.Value, self.allocator, buffered_event.data, .{}) catch continue;
                defer parsed.deinit();

                const root = parsed.value;
                const U_val = root.object.get("U") orelse continue;
                const u_val = root.object.get("u") orelse continue;
                if (U_val != .integer or u_val != .integer) continue;

                const first_update_id = U_val.integer;
                const last_update_id = u_val.integer;

                // Step 5: Discard events where u <= lastUpdateId
                if (last_update_id <= snapshot.lastUpdateId) continue;

                // Step 5: First valid event should have lastUpdateId within [U;u] range
                if (snapshot.lastUpdateId >= first_update_id and snapshot.lastUpdateId <= last_update_id) {
                    try valid_events.append(i);
                } else if (valid_events.items.len > 0) {
                    try valid_events.append(i);
                }
            }

            // Step 6: Set local order book to snapshot
            try self.applySnapshot(symbol, snapshot);

            // Step 7: Apply buffered events
            for (valid_events.items) |i| {
                const buffered_event = buffer.items[i];
                const parsed = json.parseFromSlice(json.Value, std.heap.page_allocator, buffered_event.data, .{}) catch continue;
                defer parsed.deinit();

                const root = parsed.value;
                const U_val = root.object.get("U").?;
                const u_val = root.object.get("u").?;

                try self.applyDepthUpdate(symbol, root, U_val.integer, u_val.integer);
            }

            // Clean up buffer
            for (buffer.items) |buffered_event| {
                self.allocator.free(buffered_event.data);
            }
            buffer.deinit();
            _ = self.depth_event_buffers.remove(symbol);
        }
        self.mutex.lock();
        defer self.mutex.unlock();
        const temp_symbol = try self.allocator.dupe(u8, symbol);
        try self.orderbook_initialized.put(temp_symbol, true);
    }

    fn getDepthSnapshot(self: *DepthHandler, symbol: []const u8) !DepthSnapshot {
        const symbol_upper = try std.ascii.allocUpperString(self.allocator, symbol);
        defer self.allocator.free(symbol_upper);
        const url = try std.fmt.allocPrint(self.allocator, DEPTH_API, .{symbol});
        defer self.allocator.free(url);

        const uri = try std.Uri.parse(url);
        var req = try self.http_client.open(.GET, uri, .{
            .server_header_buffer = try self.allocator.alloc(u8, 8192),
        });
        defer req.deinit();

        try req.send();
        try req.wait();

        if (req.response.status != .ok) {
            return error.DepthRequestFailed;
        }

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        const root = parsed.value;
        const last_update_id = root.object.get("lastUpdateId").?.integer;
        const bids_array = root.object.get("bids").?.array;
        const asks_array = root.object.get("asks").?.array;

        var bids_data = try self.allocator.alloc([]const u8, bids_array.items.len * 2);
        var bids_slices = try self.allocator.alloc([]const []const u8, bids_array.items.len);

        for (bids_array.items, 0..) |bid, i| {
            if (bid != .array or bid.array.items.len < 2) return error.InvalidBidFormat;
            // Duplicate the strings to ensure they outlive the parsed JSON
            const price = try self.allocator.dupe(u8, bid.array.items[0].string);
            const qty = try self.allocator.dupe(u8, bid.array.items[1].string);
            bids_data[i * 2] = price;
            bids_data[i * 2 + 1] = qty;
            bids_slices[i] = bids_data[i * 2 .. i * 2 + 2];
        }

        var asks_data = try self.allocator.alloc([]const u8, asks_array.items.len * 2);
        var asks_slices = try self.allocator.alloc([]const []const u8, asks_array.items.len);

        for (asks_array.items, 0..) |ask, i| {
            if (ask != .array or ask.array.items.len < 2) return error.InvalidAskFormat;
            // Duplicate the strings to ensure they outlive the parsed JSON
            const price = try self.allocator.dupe(u8, ask.array.items[0].string);
            const qty = try self.allocator.dupe(u8, ask.array.items[1].string);
            asks_data[i * 2] = price;
            asks_data[i * 2 + 1] = qty;
            asks_slices[i] = asks_data[i * 2 .. i * 2 + 2];
        }

        return DepthSnapshot{
            .lastUpdateId = last_update_id,
            .bids = bids_slices,
            .asks = asks_slices,
            .bids_data = bids_data,
            .asks_data = asks_data,
        };
    }

    fn freeDepthSnapshot(self: *DepthHandler, snapshot: DepthSnapshot) void {
        self.allocator.free(snapshot.bids_data);
        self.allocator.free(snapshot.asks_data);
        self.allocator.free(snapshot.bids);
        self.allocator.free(snapshot.asks);
    }

    fn applySnapshot(self: *DepthHandler, symbol: []const u8, snapshot: DepthSnapshot) !void {
        if (self.symbol_map.getPtr(symbol)) |sym| {
            sym.orderbook = OrderBook.init();

            var bid_i: usize = 0;
            while (bid_i < snapshot.bids.len and bid_i < 10) : (bid_i += 1) {
                const price = std.fmt.parseFloat(f64, snapshot.bids[bid_i][0]) catch continue;
                const quantity = std.fmt.parseFloat(f64, snapshot.bids[bid_i][1]) catch continue;
                sym.orderbook.bids[bid_i] = .{ .price = price, .quantity = quantity };
            }
            sym.orderbook.bid_count = bid_i;

            var ask_i: usize = 0;
            while (ask_i < snapshot.asks.len and ask_i < 10) : (ask_i += 1) {
                const price = std.fmt.parseFloat(f64, snapshot.asks[ask_i][0]) catch continue;
                const quantity = std.fmt.parseFloat(f64, snapshot.asks[ask_i][1]) catch continue;
                sym.orderbook.asks[ask_i] = .{ .price = price, .quantity = quantity };
            }
            sym.orderbook.ask_count = ask_i;

            sym.orderbook.last_update_id = snapshot.lastUpdateId;
            self.mutex.lock();
            defer self.mutex.unlock();
            const temp_sym = try self.allocator.dupe(u8, symbol);
            try self.last_update_ids.put(temp_sym, snapshot.lastUpdateId);
        } else {
            std.log.warn("Symbol {s} not found in symbol_map", .{symbol});
        }
    }

    fn applyDepthUpdate(self: *DepthHandler, symbol: []const u8, root: json.Value, first_update_id: i64, last_update_id: i64) !void {
        const current_update_id = self.last_update_ids.get(symbol) orelse {
            std.log.warn("Order book for {s} not initialized", .{symbol});
            return;
        };

        if (first_update_id > current_update_id + 1) {
            std.log.warn("Gap detected for {s}. Reinitializing...", .{symbol});
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.orderbook_initialized.put(symbol, false);
            try self.initializeOrderBook(symbol);
            return;
        }

        if (last_update_id <= current_update_id) {
            std.log.info("Ignoring outdated update for {s}: u={d} <= current={d}", .{ symbol, last_update_id, current_update_id });
            return;
        }

        const bids_array = root.object.get("b").?.array;
        const asks_array = root.object.get("a").?.array;

        if (self.symbol_map.getPtr(symbol)) |sym| {
            for (bids_array.items) |bid| {
                const price = std.fmt.parseFloat(f64, bid.array.items[0].string) catch continue;
                const quantity = std.fmt.parseFloat(f64, bid.array.items[1].string) catch continue;
                sym.orderbook.updateLevel(price, quantity, true);
            }

            for (asks_array.items) |ask| {
                const price = std.fmt.parseFloat(f64, ask.array.items[0].string) catch continue;
                const quantity = std.fmt.parseFloat(f64, ask.array.items[1].string) catch continue;
                sym.orderbook.updateLevel(price, quantity, false);
            }

            sym.orderbook.last_update_id = last_update_id;
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.last_update_ids.put(symbol, last_update_id);
        } else {
            std.log.warn("Symbol {s} not found in symbol_map", .{symbol});
        }
    }
};
