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

pub const MetricsChannel = struct {
    allocator: std.mem.Allocator,
    queue: std.fifo.LinearFifo(MetricData, .Dynamic),
    mutex: std.Thread.Mutex,
    condition: std.Thread.Condition,
    should_stop: std.atomic.Value(bool),
    ref_count: std.atomic.Value(u32), // Add reference counting

    pub fn init(allocator: std.mem.Allocator) !*MetricsChannel {
        const channel = try allocator.create(MetricsChannel);
        channel.* = MetricsChannel{
            .allocator = allocator,
            .queue = std.fifo.LinearFifo(MetricData, .Dynamic).init(allocator),
            .mutex = std.Thread.Mutex{},
            .condition = std.Thread.Condition{},
            .should_stop = std.atomic.Value(bool).init(false),
            .ref_count = std.atomic.Value(u32).init(1),
        };
        return channel;
    }

    pub fn addRef(self: *MetricsChannel) void {
        _ = self.ref_count.fetchAdd(1, .acq_rel);
    }

    pub fn release(self: *MetricsChannel) void {
        const old_count = self.ref_count.fetchSub(1, .acq_rel);
        if (old_count == 1) {
            // Last reference, safe to deallocate
            self.queue.deinit();
            const allocator = self.allocator;
            allocator.destroy(self);
        }
    }

    pub fn deinit(self: *MetricsChannel) void {
        self.release();
    }

    pub fn send(self: *MetricsChannel, metric: MetricData) !void {
        // Check if channel is still valid
        if (self.should_stop.load(.acquire)) {
            return; // Channel is shutting down
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Double-check after acquiring lock
        if (self.should_stop.load(.acquire)) {
            return;
        }

        try self.queue.writeItem(metric);
        self.condition.signal();
    }

    pub fn receive(self: *MetricsChannel) ?MetricData {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.queue.readItem();
    }

    pub fn stop(self: *MetricsChannel) void {
        self.should_stop.store(true, .release);
        self.condition.broadcast();
    }

    pub fn shouldStop(self: *MetricsChannel) bool {
        return self.should_stop.load(.acquire);
    }

    pub fn waitForData(self: *MetricsChannel) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.queue.count == 0 and !self.shouldStop()) {
            self.condition.wait(&self.mutex);
        }
    }
};

pub const MetricsCollector = struct {
    depth_msg_count: std.atomic.Value(u64),
    ticker_msg_count: std.atomic.Value(u64),
    last_reset_time: std.atomic.Value(i64),
    channel: *MetricsChannel,

    pub fn init(channel: *MetricsChannel) MetricsCollector {
        channel.addRef(); // Add reference for this collector
        return MetricsCollector{
            .depth_msg_count = std.atomic.Value(u64).init(0),
            .ticker_msg_count = std.atomic.Value(u64).init(0),
            .last_reset_time = std.atomic.Value(i64).init(std.time.milliTimestamp()),
            .channel = channel,
        };
    }

    pub fn deinit(self: *MetricsCollector) void {
        self.channel.release(); // Release reference
    }

    pub fn recordDepthMessage(self: *MetricsCollector, duration_us: f64) void {
        _ = self.depth_msg_count.fetchAdd(1, .acq_rel);

        const duration_metric = MetricData{
            .metric_type = .depth_handler_duration,
            .value = duration_us,
            .timestamp = std.time.milliTimestamp(),
        };
        self.channel.send(duration_metric) catch |err| {
            // Log error but don't crash
            std.debug.print("Failed to send depth duration metric: {}\n", .{err});
        };

        self.checkAndResetCounters();
    }

    pub fn recordTickerMessage(self: *MetricsCollector, duration_us: f64) void {
        _ = self.ticker_msg_count.fetchAdd(1, .acq_rel);

        const duration_metric = MetricData{
            .metric_type = .ticker_handler_duration,
            .value = duration_us,
            .timestamp = std.time.milliTimestamp(),
        };
        self.channel.send(duration_metric) catch |err| {
            // Log error but don't crash
            std.debug.print("Failed to send ticker duration metric: {}\n", .{err});
        };

        self.checkAndResetCounters();
    }

    fn checkAndResetCounters(self: *MetricsCollector) void {
        const current_time = std.time.milliTimestamp();
        const last_reset = self.last_reset_time.load(.acquire);
        const elapsed_ms = current_time - last_reset;

        if (elapsed_ms >= 1000) {
            // Try to atomically update the reset time
            if (self.last_reset_time.cmpxchgWeak(last_reset, current_time, .acq_rel, .acquire)) |_| {
                // Another thread already reset, skip
                return;
            }

            // Get and reset counters atomically
            const depth_count = self.depth_msg_count.swap(0, .acq_rel);
            const ticker_count = self.ticker_msg_count.swap(0, .acq_rel);

            // Send message count metrics
            if (depth_count > 0) {
                const depth_rate = @as(f64, @floatFromInt(depth_count)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
                const depth_metric = MetricData{
                    .metric_type = .depth_handler_msg,
                    .value = depth_rate,
                    .timestamp = current_time,
                };
                self.channel.send(depth_metric) catch |err| {
                    std.debug.print("Failed to send depth rate metric: {}\n", .{err});
                };
            }

            if (ticker_count > 0) {
                const ticker_rate = @as(f64, @floatFromInt(ticker_count)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
                const ticker_metric = MetricData{
                    .metric_type = .ticker_handler_msg,
                    .value = ticker_rate,
                    .timestamp = current_time,
                };
                self.channel.send(ticker_metric) catch |err| {
                    std.debug.print("Failed to send ticker rate metric: {}\n", .{err});
                };
            }
        }
    }
};

pub fn metricsThread(channel: *MetricsChannel) void {
    channel.addRef(); // Add reference for this thread
    defer channel.release(); // Release when thread exits

    std.debug.print("Metrics thread started\n", .{});

    while (!channel.shouldStop()) {
        channel.waitForData();

        // Process all available metrics
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
    }

    std.debug.print("Metrics thread stopped\n", .{});
}
