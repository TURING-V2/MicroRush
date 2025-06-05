const std = @import("std");

pub const MetricType = enum {
    depth_handler_msg,
    ticker_handler_msg,
    depth_handler_duration,
    ticker_handler_duration,
};

pub const MetricData = struct {
    metric_type: MetricType,
    value: f64,
    timestamp: i64,
};

pub const SimpleLockFreeRingBuffer = struct {
    buffer: []MetricData,
    head: std.atomic.Value(u64),
    tail: std.atomic.Value(u64),
    capacity: u64,

    pub fn init(allocator: std.mem.Allocator, size: u64) !SimpleLockFreeRingBuffer {
        const capacity = std.math.ceilPowerOfTwo(u64, size) catch return error.TooLarge;
        const buffer = try allocator.alloc(MetricData, capacity);
        @memset(buffer, std.mem.zeroes(MetricData));

        return SimpleLockFreeRingBuffer{
            .buffer = buffer,
            .head = std.atomic.Value(u64).init(0),
            .tail = std.atomic.Value(u64).init(0),
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *SimpleLockFreeRingBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
    }

    pub fn push(self: *SimpleLockFreeRingBuffer, item: MetricData) bool {
        const current_tail = self.tail.load(.unordered);
        const next_tail = (current_tail + 1) & (self.capacity - 1);
        const current_head = self.head.load(.acquire);

        if (next_tail == current_head) {
            return false; // buffer full
        }

        self.buffer[current_tail] = item;
        self.tail.store(next_tail, .release);
        return true;
    }

    pub fn pop(self: *SimpleLockFreeRingBuffer) ?MetricData {
        const current_head = self.head.load(.unordered);
        const current_tail = self.tail.load(.acquire);

        if (current_head == current_tail) {
            return null; // buffer empty
        }

        const item = self.buffer[current_head];
        const next_head = (current_head + 1) & (self.capacity - 1);
        self.head.store(next_head, .release);

        return item;
    }

    pub fn isEmpty(self: *SimpleLockFreeRingBuffer) bool {
        return self.head.load(.acquire) == self.tail.load(.acquire);
    }
};

// lock-free ring buffer using atomic operations
pub const LockFreeRingBuffer = struct {
    buffer: []MetricData,
    head: std.atomic.Value(u64),
    tail: std.atomic.Value(u64),
    capacity: u64,

    pub fn init(allocator: std.mem.Allocator, size: u64) !LockFreeRingBuffer {
        const capacity = std.math.ceilPowerOfTwo(u64, size) catch return error.TooLarge;
        const buffer = try allocator.alloc(MetricData, capacity);
        @memset(buffer, std.mem.zeroes(MetricData));

        return LockFreeRingBuffer{
            .buffer = buffer,
            .head = std.atomic.Value(u64).init(0),
            .tail = std.atomic.Value(u64).init(0),
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *LockFreeRingBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
    }

    pub fn push(self: *LockFreeRingBuffer, item: MetricData) bool {
        var retries: u32 = 0;
        while (retries < 1000) : (retries += 1) {
            const current_tail = self.tail.load(.acquire);
            const next_tail = (current_tail + 1) & (self.capacity - 1);
            const current_head = self.head.load(.acquire);

            if (next_tail == current_head) {
                return false; // buffer full
            }

            if (self.tail.cmpxchgWeak(current_tail, next_tail, .acq_rel, .acquire)) |_| {
                continue; // CAS failed, retry
            }

            self.buffer[current_tail] = item;
            return true;
        }
        return false; // too many retries, likely contention issue
    }

    pub fn pop(self: *LockFreeRingBuffer) ?MetricData {
        var retries: u32 = 0;
        while (retries < 1000) : (retries += 1) {
            const current_head = self.head.load(.acquire);
            const current_tail = self.tail.load(.acquire);

            if (current_head == current_tail) {
                return null;
            }

            const item = self.buffer[current_head];
            const next_head = (current_head + 1) & (self.capacity - 1);

            if (self.head.cmpxchgWeak(current_head, next_head, .acq_rel, .acquire)) |_| {
                continue; // CAS failed, retry
            }

            return item;
        }
        return null; // too many retries
    }

    pub fn isEmpty(self: *LockFreeRingBuffer) bool {
        return self.head.load(.acquire) == self.tail.load(.acquire);
    }
};

pub const MetricsChannel = struct {
    allocator: std.mem.Allocator,
    ring_buffer: SimpleLockFreeRingBuffer,
    should_stop: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator) !*MetricsChannel {
        const channel = try allocator.create(MetricsChannel);
        const ring_buffer = try SimpleLockFreeRingBuffer.init(allocator, 90000);

        channel.* = MetricsChannel{
            .allocator = allocator,
            .ring_buffer = ring_buffer,
            .should_stop = std.atomic.Value(bool).init(false),
        };
        return channel;
    }

    pub fn deinit(self: *MetricsChannel) void {
        self.ring_buffer.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn send(self: *MetricsChannel, metric: MetricData) bool {
        return self.ring_buffer.push(metric);
    }

    pub fn receive(self: *MetricsChannel) ?MetricData {
        return self.ring_buffer.pop();
    }

    pub fn stop(self: *MetricsChannel) void {
        self.should_stop.store(true, .release);
    }

    pub fn shouldStop(self: *MetricsChannel) bool {
        return self.should_stop.load(.acquire);
    }

    pub fn hasData(self: *MetricsChannel) bool {
        return !self.ring_buffer.isEmpty();
    }
};

pub const MetricsCollector = struct {
    depth_msg_count: std.atomic.Value(u64),
    ticker_msg_count: std.atomic.Value(u64),
    last_reset_time: std.atomic.Value(i64),
    channel: *MetricsChannel,

    pub fn init(channel: *MetricsChannel) MetricsCollector {
        return MetricsCollector{
            .depth_msg_count = std.atomic.Value(u64).init(0),
            .ticker_msg_count = std.atomic.Value(u64).init(0),
            .last_reset_time = std.atomic.Value(i64).init(std.time.milliTimestamp()),
            .channel = channel,
        };
    }

    pub fn recordDepthMessage(self: *MetricsCollector, duration_us: f64) void {
        _ = self.depth_msg_count.fetchAdd(1, .acq_rel);

        const duration_metric = MetricData{
            .metric_type = .depth_handler_duration,
            .value = duration_us,
            .timestamp = std.time.milliTimestamp(),
        };
        _ = self.channel.send(duration_metric);

        self.checkAndResetCounters();
    }

    pub fn recordTickerMessage(self: *MetricsCollector, duration_us: f64) void {
        _ = self.ticker_msg_count.fetchAdd(1, .acq_rel);

        const duration_metric = MetricData{
            .metric_type = .ticker_handler_duration,
            .value = duration_us,
            .timestamp = std.time.milliTimestamp(),
        };
        _ = self.channel.send(duration_metric);

        self.checkAndResetCounters();
    }

    fn checkAndResetCounters(self: *MetricsCollector) void {
        const current_time = std.time.milliTimestamp();
        const last_reset = self.last_reset_time.load(.acquire);
        const elapsed_ms = current_time - last_reset;

        if (elapsed_ms >= 1000) {
            // try to atomically update the reset time to avoid multiple threads doing this
            if (self.last_reset_time.cmpxchgWeak(last_reset, current_time, .acq_rel, .acquire) == null) {
                const depth_count = self.depth_msg_count.swap(0, .acq_rel);
                const ticker_count = self.ticker_msg_count.swap(0, .acq_rel);

                const actual_elapsed_ms = current_time - last_reset;
                const elapsed_seconds = @as(f64, @floatFromInt(actual_elapsed_ms)) / 1000.0;

                if (depth_count > 0) {
                    const depth_rate = @as(f64, @floatFromInt(depth_count)) / elapsed_seconds;
                    const depth_metric = MetricData{
                        .metric_type = .depth_handler_msg,
                        .value = depth_rate,
                        .timestamp = current_time,
                    };
                    _ = self.channel.send(depth_metric);
                }

                if (ticker_count > 0) {
                    const ticker_rate = @as(f64, @floatFromInt(ticker_count)) / elapsed_seconds;
                    const ticker_metric = MetricData{
                        .metric_type = .ticker_handler_msg,
                        .value = ticker_rate,
                        .timestamp = current_time,
                    };
                    _ = self.channel.send(ticker_metric);
                }
            }
        }
    }
};

pub fn metricsThread(channel: *MetricsChannel) void {
    std.debug.print("Metrics thread started\n", .{});
    while (!channel.shouldStop()) {
        while (channel.receive()) |metric| {
            switch (metric.metric_type) {
                .depth_handler_msg => {
                    std.debug.print("DepthHandler messages per second: {d:.2}\n", .{metric.value});
                },
                .ticker_handler_msg => {
                    std.debug.print("TickerHandler messages per second: {d:.2}\n", .{metric.value});
                },
                .depth_handler_duration => {
                    std.debug.print("DepthHandler.serverMessage took: {d:.2} μs\n", .{metric.value});
                },
                .ticker_handler_duration => {
                    std.debug.print("TickerHandler.serverMessage took: {d:.2} μs\n", .{metric.value});
                },
            }
        }

        if (!channel.hasData()) {
            std.time.sleep(1000000);
        }
    }

    std.debug.print("Metrics thread stopped\n", .{});
}
