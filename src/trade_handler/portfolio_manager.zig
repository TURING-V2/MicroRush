const std = @import("std");
const types = @import("../types.zig");
const symbol_map = @import("../symbol-map.zig");
const SymbolMap = symbol_map.SymbolMap;
const TradingSignal = types.TradingSignal;
const SignalType = types.SignalType;

const Trade = struct {
    symbol: []const u8,
    side: SignalType,
    amount: f64,
    price: f64,
    fee: f64,
    timestamp: i128,
    rsi_value: f32,
    orderbook_percentage: f32,
};

const PortfolioPosition = struct {
    symbol: []const u8,
    amount: f64,
    avg_entry_price: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
    entry_timestamp: i128,
    entry_rsi: f32,
    is_open: bool,
};

pub const PortfolioManager = struct {
    allocator: std.mem.Allocator,
    symbol_map: *const SymbolMap,

    initial_balance: f64,
    current_balance: f64,
    max_positions: usize,
    position_size_usdt: f64,
    fee_rate: f64,

    positions: std.StringHashMap(PortfolioPosition),
    trade_history: std.ArrayList(Trade),

    // P&L tracking
    total_realized_pnl: f64,
    total_unrealized_pnl: f64,
    total_trades: usize,
    winning_trades: usize,

    // binance testnet connection (mock for now)
    testnet_enabled: bool,

    mutex: std.Thread.Mutex,

    // Performance tracking
    last_balance_log: i128,
    balance_log_interval: i128,

    pub fn init(allocator: std.mem.Allocator, sym_map: *const SymbolMap) PortfolioManager {
        return PortfolioManager{
            .allocator = allocator,
            .symbol_map = sym_map,
            .initial_balance = 100.0,
            .current_balance = 100.0,
            .max_positions = 10,
            .position_size_usdt = 100.0 / @as(f64, @floatFromInt(10)),
            .fee_rate = 0.001, // 0.1%
            .positions = std.StringHashMap(PortfolioPosition).init(allocator),
            .trade_history = std.ArrayList(Trade).init(allocator),
            .total_realized_pnl = 0.0,
            .total_unrealized_pnl = 0.0,
            .total_trades = 0,
            .winning_trades = 0,
            .testnet_enabled = true,
            .mutex = std.Thread.Mutex{},
            .last_balance_log = std.time.nanoTimestamp(),
            .balance_log_interval = 60_000_000_000,
        };
    }

    pub fn deinit(self: *PortfolioManager) void {
        self.positions.deinit();
        self.trade_history.deinit();
    }

    pub fn processSignal(self: *PortfolioManager, signal: TradingSignal) !void {
        const current_time = std.time.nanoTimestamp();
        const last_close_price = try symbol_map.getLastClosePrice(self.symbol_map, signal.symbol_name);
        switch (signal.signal_type) {
            .BUY => self.executeBuy(signal, last_close_price, current_time),
            .SELL => self.executeSell(signal, last_close_price, current_time),
            .HOLD => {},
        }
        try self.updateUnrealizedPnL();
        if (current_time - self.last_balance_log >= self.balance_log_interval) {
            self.logPerformance();
            self.last_balance_log = current_time;
        }
    }

    fn executeBuy(self: *PortfolioManager, signal: TradingSignal, price: f64, timestamp: i128) void {
        if (self.positions.contains(signal.symbol_name)) {
            std.log.warn("Already have position in {s}, skipping BUY", .{signal.symbol_name});
            return;
        }
        if (self.getOpenPositionsCount() >= self.max_positions) {
            std.log.warn("Max positions reached, skipping BUY for {s}", .{signal.symbol_name});
            return;
        }
        if (self.current_balance < self.position_size_usdt) {
            std.log.warn("Insufficient balance for BUY {s}", .{signal.symbol_name});
            return;
        }

        const amount = self.position_size_usdt / (price * (1.0 + self.fee_rate));
        const trade_volume = amount * price;
        const fee = trade_volume * self.fee_rate;

        // trade record
        const trade = Trade{
            .symbol = signal.symbol_name,
            .side = .BUY,
            .amount = amount,
            .price = price,
            .fee = fee,
            .timestamp = timestamp,
            .rsi_value = signal.rsi_value,
            .orderbook_percentage = signal.orderbook_percentage,
        };

        // create position
        const position = PortfolioPosition{
            .symbol = signal.symbol_name,
            .amount = amount,
            .avg_entry_price = price,
            .unrealized_pnl = 0.0,
            .realized_pnl = 0.0,
            .entry_timestamp = timestamp,
            .entry_rsi = signal.rsi_value,
            .is_open = true,
        };

        // update portfolio
        self.positions.put(signal.symbol_name, position) catch |err| {
            std.log.err("Failed to add position: {}", .{err});
            return;
        };

        self.trade_history.append(trade) catch |err| {
            std.log.err("Failed to record trade: {}", .{err});
        };

        self.current_balance -= self.position_size_usdt;
        self.total_trades += 1;

        // Updated log to include the fee
        std.log.info("EXECUTED BUY: {s} | Amount: {d:.6} | Price: ${d:.4} | Size: ${d:.2} | Fee: ${d:.4} | RSI: {d:.2}", .{
            signal.symbol_name, amount, price, self.position_size_usdt, fee, signal.rsi_value,
        });

        if (self.testnet_enabled) {
            std.log.info("Testnet: Would place BUY order for {s}", .{signal.symbol_name});
        }
    }

    fn executeSell(self: *PortfolioManager, signal: TradingSignal, price: f64, timestamp: i128) void {
        if (self.positions.getPtr(signal.symbol_name)) |position| {
            if (!position.is_open) {
                std.log.warn("Position for {s} is already closed", .{signal.symbol_name});
                return;
            }

            const trade_volume = position.amount * price;
            const fee = trade_volume * self.fee_rate;
            const net_proceeds = trade_volume - fee;

            const pnl = net_proceeds - self.position_size_usdt; // P&L is based on net proceeds vs initial cost
            const pnl_percentage = (pnl / self.position_size_usdt) * 100.0;

            // trade record
            const trade = Trade{
                .symbol = signal.symbol_name,
                .side = .SELL,
                .amount = position.amount,
                .price = price,
                .fee = fee,
                .timestamp = timestamp,
                .rsi_value = signal.rsi_value,
                .orderbook_percentage = signal.orderbook_percentage,
            };

            // update position
            position.is_open = false;
            position.realized_pnl = pnl;
            position.unrealized_pnl = 0.0;

            // update portfolio
            self.trade_history.append(trade) catch |err| {
                std.log.err("Failed to record trade: {}", .{err});
            };

            self.current_balance += net_proceeds; // add net proceeds to balance
            self.total_realized_pnl += pnl;
            self.total_trades += 1;

            if (pnl > 0) {
                self.winning_trades += 1;
            }

            const hold_duration_ms = @divFloor(timestamp - position.entry_timestamp, 1_000_000);

            // Updated log to include fee and show net P&L
            std.log.info("EXECUTED SELL: {s} | Amount: {d:.6} | Price: ${d:.4} | P&L: ${d:.2} ({d:.1}%) | Fee: ${d:.4} | Hold: {d}ms | Entry RSI: {d:.2} | Exit RSI: {d:.2}", .{
                signal.symbol_name, position.amount, price, pnl, pnl_percentage, fee, hold_duration_ms, position.entry_rsi, signal.rsi_value,
            });

            if (self.testnet_enabled) {
                std.log.info("Testnet: Would place SELL order for {s}", .{signal.symbol_name});
            }
        } else {
            std.log.warn("No position found for SELL signal: {s}", .{signal.symbol_name});
        }
    }

    fn updateUnrealizedPnL(self: *PortfolioManager) !void {
        self.total_unrealized_pnl = 0.0;

        var iterator = self.positions.iterator();
        while (iterator.next()) |entry| {
            const position = entry.value_ptr;
            if (position.is_open) {
                const current_price = try symbol_map.getLastClosePrice(self.symbol_map, position.symbol);

                // the net value if we were to sell right now
                const gross_current_value = current_price * position.amount;
                const estimated_sell_fee = gross_current_value * self.fee_rate;
                const net_current_value = gross_current_value - estimated_sell_fee;

                // unrealized P&L is the net value minus the initial total cost
                const unrealized_pnl = net_current_value - self.position_size_usdt;

                position.unrealized_pnl = unrealized_pnl;
                self.total_unrealized_pnl += unrealized_pnl;
            }
        }
    }

    fn getOpenPositionsCount(self: *PortfolioManager) usize {
        var count: usize = 0;
        var iterator = self.positions.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.is_open) {
                count += 1;
            }
        }
        return count;
    }

    pub fn logPerformance(self: *PortfolioManager) void {
        const total_pnl = self.total_realized_pnl + self.total_unrealized_pnl;
        const total_return_pct = (total_pnl / self.initial_balance) * 100.0;
        const win_rate = if (self.total_trades > 0)
            (@as(f64, @floatFromInt(self.winning_trades)) / @as(f64, @floatFromInt(self.total_trades / 2))) * 100.0 // A full trade is buy+sell
        else
            0.0;

        const open_positions = self.getOpenPositionsCount();
        const closed_trades = (self.total_trades - open_positions) / 2;

        std.log.info("=== PORTFOLIO PERFORMANCE ===", .{});
        std.log.info("Balance: ${d:.2} | Initial: ${d:.2} | Total Net P&L: ${d:.2} ({d:.1}%)", .{
            self.current_balance + self.total_unrealized_pnl, self.initial_balance, total_pnl, total_return_pct,
        });
        std.log.info("Realized P&L: ${d:.2} | Unrealized P&L: ${d:.2}", .{
            self.total_realized_pnl, self.total_unrealized_pnl,
        });
        std.log.info("Closed Trades: {} | Winning Trades: {} | Win Rate: {d:.1}%", .{
            closed_trades, self.winning_trades, win_rate,
        });
        std.log.info("Open Positions: {} / {} | Available Balance: ${d:.2}", .{
            open_positions, self.max_positions, self.current_balance,
        });
        std.log.info("=============================", .{});
    }

    pub fn getStats(self: *PortfolioManager) struct {
        balance: f64,
        total_pnl: f64,
        realized_pnl: f64,
        unrealized_pnl: f64,
        total_trades: usize,
        win_rate: f64,
        open_positions: usize,
    } {
        const total_pnl = self.total_realized_pnl + self.total_unrealized_pnl;
        const closed_trades = (self.total_trades - self.getOpenPositionsCount()) / 2;
        const win_rate = if (closed_trades > 0)
            (@as(f64, @floatFromInt(self.winning_trades)) / @as(f64, @floatFromInt(closed_trades))) * 100.0
        else
            0.0;
        return .{
            .balance = self.current_balance,
            .total_pnl = total_pnl,
            .realized_pnl = self.total_realized_pnl,
            .unrealized_pnl = self.total_unrealized_pnl,
            .total_trades = closed_trades,
            .win_rate = win_rate,
            .open_positions = self.getOpenPositionsCount(),
        };
    }
};
