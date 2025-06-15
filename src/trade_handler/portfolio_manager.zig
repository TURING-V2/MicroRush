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
    signal_strength: f32,
    position_size_usdt: f64,
};

const PortfolioPosition = struct {
    symbol: []const u8,
    amount: f64,
    avg_entry_price: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
    entry_timestamp: i128,
    entry_rsi: f32,
    entry_signal_strength: f32,
    position_size_usdt: f64,
    is_open: bool,
};

pub const PortfolioManager = struct {
    allocator: std.mem.Allocator,
    symbol_map: *const SymbolMap,

    initial_balance: f64,
    current_balance: f64,
    max_positions: usize,

    base_position_size_usdt: f64,
    max_position_size_usdt: f64,
    min_position_size_usdt: f64,
    max_portfolio_risk_pct: f64,

    fee_rate: f64,

    positions: std.StringHashMap(PortfolioPosition),
    trade_history: std.ArrayList(Trade),

    // P&L tracking
    total_realized_pnl: f64,
    total_unrealized_pnl: f64,
    total_trades: usize,
    winning_trades: usize,
    losing_trades: usize,

    // Profit/Loss statistics
    total_profit_pct: f64,
    total_loss_pct: f64,
    highest_profit_pct: f64,
    lowest_loss_pct: f64,

    // Signal strength statistics
    strong_signal_trades: usize,
    strong_signal_wins: usize,
    strong_signal_losses: usize,
    strong_signal_total_profit_pct: f64,
    strong_signal_total_loss_pct: f64,
    strong_signal_highest_profit_pct: f64,
    strong_signal_lowest_loss_pct: f64,

    normal_signal_trades: usize,
    normal_signal_wins: usize,
    normal_signal_losses: usize,
    normal_signal_total_profit_pct: f64,
    normal_signal_total_loss_pct: f64,
    normal_signal_highest_profit_pct: f64,
    normal_signal_lowest_loss_pct: f64,

    // binance testnet connection (mock for now)
    testnet_enabled: bool,

    mutex: std.Thread.Mutex,

    last_balance_log: i128,
    balance_log_interval: i128,

    // Stop loss configuration
    stop_loss_percentage: f64, // e.g., 0.02 for 2% stop loss

    pub fn init(allocator: std.mem.Allocator, sym_map: *const SymbolMap) PortfolioManager {
        const initial_balance = 1000.0;
        const base_size = initial_balance / 20.0; // 5% base position

        return PortfolioManager{
            .allocator = allocator,
            .symbol_map = sym_map,
            .initial_balance = initial_balance,
            .current_balance = initial_balance,
            .max_positions = 15,

            // Dynamic sizing parameters
            .base_position_size_usdt = base_size,
            .max_position_size_usdt = initial_balance / 10.0, // Max 10% per trade
            .min_position_size_usdt = initial_balance / 50.0, // Min 2% per trade
            .max_portfolio_risk_pct = 0.15, // Max 15% of portfolio at risk

            .fee_rate = 0.001, // 0.1%
            .positions = std.StringHashMap(PortfolioPosition).init(allocator),
            .trade_history = std.ArrayList(Trade).init(allocator),
            .total_realized_pnl = 0.0,
            .total_unrealized_pnl = 0.0,
            .total_trades = 0,
            .winning_trades = 0,
            .losing_trades = 0,

            // P&L statistics
            .total_profit_pct = 0.0,
            .total_loss_pct = 0.0,
            .highest_profit_pct = 0.0,
            .lowest_loss_pct = 0.0,

            // Strong signal statistics
            .strong_signal_trades = 0,
            .strong_signal_wins = 0,
            .strong_signal_losses = 0,
            .strong_signal_total_profit_pct = 0.0,
            .strong_signal_total_loss_pct = 0.0,
            .strong_signal_highest_profit_pct = 0.0,
            .strong_signal_lowest_loss_pct = 0.0,

            // Normal signal statistics
            .normal_signal_trades = 0,
            .normal_signal_wins = 0,
            .normal_signal_losses = 0,
            .normal_signal_total_profit_pct = 0.0,
            .normal_signal_total_loss_pct = 0.0,
            .normal_signal_highest_profit_pct = 0.0,
            .normal_signal_lowest_loss_pct = 0.0,

            .testnet_enabled = true,
            .mutex = std.Thread.Mutex{},
            .last_balance_log = std.time.nanoTimestamp(),
            .balance_log_interval = 30_000_000_000, // 30 seconds for HFT
            .stop_loss_percentage = 0.01,
        };
    }

    pub fn deinit(self: *PortfolioManager) void {
        self.positions.deinit();
        self.trade_history.deinit();
    }

    pub fn calculatePositionSize(self: *const PortfolioManager, signal_strength: f32, available_liquidity: f64, symbol_price: f64) f32 {
        const base_size = self.base_position_size_usdt * signal_strength;

        // Adjust based on liquidity
        const max_liquidity_consumption = 0.25; // Max 25% of available liquidity
        const available_size = available_liquidity * symbol_price * max_liquidity_consumption;

        // Take minimum of intended size and liquidity-constrained size
        const liquidity_adjusted = @min(base_size, available_size);

        return @as(f32, @floatCast(@max(self.min_position_size_usdt, @min(liquidity_adjusted, self.max_position_size_usdt))));
    }

    pub fn checkStopLossConditions(self: *PortfolioManager) !void {
        var positions_to_close = std.ArrayList([]const u8).init(self.allocator);
        defer positions_to_close.deinit();

        const current_time = std.time.nanoTimestamp();
        var iterator = self.positions.iterator();

        while (iterator.next()) |entry| {
            const position = entry.value_ptr;
            if (!position.is_open) continue;

            const current_price = symbol_map.getLastClosePrice(self.symbol_map, position.symbol) catch |err| {
                std.log.warn("Failed to get current price for {s}: {}", .{ position.symbol, err });
                continue;
            };

            var should_close = false;
            var close_reason: []const u8 = "";

            const stop_loss_price = position.avg_entry_price * (1.0 - self.stop_loss_percentage);
            if (current_price <= stop_loss_price) {
                should_close = true;
                close_reason = "STOP LOSS";
            }

            const time_elapsed = current_time - position.entry_timestamp;
            if (time_elapsed >= 1 * 60 * 1_000_000_000) { // crosses 1 minute
                should_close = true;
                close_reason = "TIME LIMIT";
            }

            const profit_percentage = ((current_price - position.avg_entry_price) / position.avg_entry_price) * 100.0;
            if (profit_percentage > 0.3) {
                should_close = true;
                close_reason = "PROFIT TARGET";
            }

            if (should_close) {
                positions_to_close.append(position.symbol) catch |err| {
                    std.log.err("Failed to add position to close list: {}", .{err});
                    continue;
                };
            }
        }

        for (positions_to_close.items) |symbol| {
            if (self.positions.getPtr(symbol)) |position| {
                const current_price = symbol_map.getLastClosePrice(self.symbol_map, symbol) catch continue;
                const current_time_final = std.time.nanoTimestamp();
                const time_elapsed = current_time_final - position.entry_timestamp;
                const profit_percentage = ((current_price - position.avg_entry_price) / position.avg_entry_price) * 100.0;

                var actual_reason: []const u8 = "STOP LOSS";
                if (time_elapsed >= 5_000_000_000) {
                    actual_reason = "TIME LIMIT";
                } else if (profit_percentage > 0.3) {
                    actual_reason = "PROFIT TARGET";
                }

                std.log.warn("{s} TRIGGERED for {s}: Entry: ${d:.4}, Current: ${d:.4}, P&L: {d:.1}%, Time: {d:.1}s", .{
                    actual_reason,
                    symbol,
                    position.avg_entry_price,
                    current_price,
                    profit_percentage,
                    @as(f64, @floatFromInt(time_elapsed)) / 1_000_000_000.0,
                });

                const sell_signal = TradingSignal{
                    .symbol_name = symbol,
                    .signal_type = .SELL,
                    .rsi_value = 50.0, // neutral RSI
                    .orderbook_percentage = 0.0,
                    .timestamp = current_time_final,
                    .signal_strength = 1.0,
                };

                self.executeSell(sell_signal, current_price, current_time_final);
            }
        }
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
            return;
        }

        if (self.getOpenPositionsCount() >= self.max_positions) {
            //std.log.warn("Max positions reached, skipping BUY for {s}", .{signal.symbol_name});
            return;
        }

        const position_size_usdt = self.calculatePositionSize(signal.signal_strength, self.current_balance, price);

        if (self.current_balance < position_size_usdt) {
            std.log.warn("Insufficient balance for BUY {s} (need ${d:.2}, have ${d:.2})", .{ signal.symbol_name, position_size_usdt, self.current_balance });
            return;
        }

        const amount = position_size_usdt / (price * (1.0 + self.fee_rate));
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
            .signal_strength = signal.signal_strength,
            .position_size_usdt = position_size_usdt,
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
            .entry_signal_strength = signal.signal_strength,
            .position_size_usdt = position_size_usdt,
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

        self.current_balance -= position_size_usdt;
        self.total_trades += 1;

        // track signal strength stats
        if (signal.signal_strength >= 0.9) {
            self.strong_signal_trades += 1;
        } else {
            self.normal_signal_trades += 1;
        }
        std.log.info("EXECUTED BUY: {s} | Amount: {d:.6} | Price: ${d:.4} | Size: ${d:.2} | Strength: {d:.2} | Fee: ${d:.4} | RSI: {d:.2}", .{
            signal.symbol_name, amount, price, position_size_usdt, signal.signal_strength, fee, signal.rsi_value,
        });
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

            const pnl = net_proceeds - position.position_size_usdt;
            const pnl_percentage = (pnl / position.position_size_usdt) * 100.0;

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
                .signal_strength = signal.signal_strength,
                .position_size_usdt = position.position_size_usdt,
            };

            // update position
            position.is_open = false;
            position.realized_pnl = pnl;
            position.unrealized_pnl = 0.0;

            // update portfolio
            self.trade_history.append(trade) catch |err| {
                std.log.err("Failed to record trade: {}", .{err});
            };

            self.current_balance += net_proceeds;
            self.total_realized_pnl += pnl;
            self.total_trades += 1;

            // Update overall statistics
            if (pnl > 0) {
                self.winning_trades += 1;
                self.total_profit_pct += pnl_percentage;
                if (pnl_percentage > self.highest_profit_pct) {
                    self.highest_profit_pct = pnl_percentage;
                }
            } else {
                self.losing_trades += 1;
                self.total_loss_pct += pnl_percentage; // This will be negative
                if (pnl_percentage < self.lowest_loss_pct) {
                    self.lowest_loss_pct = pnl_percentage;
                }
            }

            // Update signal strength specific statistics
            const is_strong_signal = position.entry_signal_strength >= 0.9;
            if (is_strong_signal) {
                if (pnl > 0) {
                    self.strong_signal_wins += 1;
                    self.strong_signal_total_profit_pct += pnl_percentage;
                    if (pnl_percentage > self.strong_signal_highest_profit_pct) {
                        self.strong_signal_highest_profit_pct = pnl_percentage;
                    }
                } else {
                    self.strong_signal_losses += 1;
                    self.strong_signal_total_loss_pct += pnl_percentage;
                    if (pnl_percentage < self.strong_signal_lowest_loss_pct) {
                        self.strong_signal_lowest_loss_pct = pnl_percentage;
                    }
                }
            } else {
                if (pnl > 0) {
                    self.normal_signal_wins += 1;
                    self.normal_signal_total_profit_pct += pnl_percentage;
                    if (pnl_percentage > self.normal_signal_highest_profit_pct) {
                        self.normal_signal_highest_profit_pct = pnl_percentage;
                    }
                } else {
                    self.normal_signal_losses += 1;
                    self.normal_signal_total_loss_pct += pnl_percentage;
                    if (pnl_percentage < self.normal_signal_lowest_loss_pct) {
                        self.normal_signal_lowest_loss_pct = pnl_percentage;
                    }
                }
            }

            const hold_duration_ms = @divFloor(timestamp - position.entry_timestamp, 1_000_000);

            std.log.info("EXECUTED SELL: {s} | Amount: {d:.6} | Price: ${d:.4} | P&L: ${d:.2} ({d:.1}%) | Entry Strength: {d:.2} | Exit Strength: {d:.2} | Hold: {d}ms", .{
                signal.symbol_name, position.amount, price, pnl, pnl_percentage, position.entry_signal_strength, signal.signal_strength, hold_duration_ms,
            });
        }
    }

    fn updateUnrealizedPnL(self: *PortfolioManager) !void {
        self.total_unrealized_pnl = 0.0;

        var iterator = self.positions.iterator();
        while (iterator.next()) |entry| {
            const position = entry.value_ptr;
            if (position.is_open) {
                const current_price = try symbol_map.getLastClosePrice(self.symbol_map, position.symbol);

                const gross_current_value = current_price * position.amount;
                const estimated_sell_fee = gross_current_value * self.fee_rate;
                const net_current_value = gross_current_value - estimated_sell_fee;

                const unrealized_pnl = net_current_value - position.position_size_usdt;

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
        const current_portfolio_value = self.current_balance + self.total_unrealized_pnl;
        const total_return_pct = (total_pnl / self.initial_balance) * 100.0;

        const closed_trades = (self.total_trades - self.getOpenPositionsCount()) / 2;
        const overall_win_rate = if (closed_trades > 0)
            (@as(f64, @floatFromInt(self.winning_trades)) / @as(f64, @floatFromInt(closed_trades))) * 100.0
        else
            0.0;
        const overall_loss_rate = if (closed_trades > 0)
            (@as(f64, @floatFromInt(self.losing_trades)) / @as(f64, @floatFromInt(closed_trades))) * 100.0
        else
            0.0;

        // Calculate average profit and loss percentages
        const avg_profit_pct = if (self.winning_trades > 0)
            self.total_profit_pct / @as(f64, @floatFromInt(self.winning_trades))
        else
            0.0;
        const avg_loss_pct = if (self.losing_trades > 0)
            self.total_loss_pct / @as(f64, @floatFromInt(self.losing_trades))
        else
            0.0;

        // Signal strength performance
        const strong_win_rate = if (self.strong_signal_trades > 0)
            (@as(f64, @floatFromInt(self.strong_signal_wins)) / @as(f64, @floatFromInt(self.strong_signal_trades))) * 100.0
        else
            0.0;
        const strong_loss_rate = if (self.strong_signal_trades > 0)
            (@as(f64, @floatFromInt(self.strong_signal_losses)) / @as(f64, @floatFromInt(self.strong_signal_trades))) * 100.0
        else
            0.0;

        const normal_win_rate = if (self.normal_signal_trades > 0)
            (@as(f64, @floatFromInt(self.normal_signal_wins)) / @as(f64, @floatFromInt(self.normal_signal_trades))) * 100.0
        else
            0.0;
        const normal_loss_rate = if (self.normal_signal_trades > 0)
            (@as(f64, @floatFromInt(self.normal_signal_losses)) / @as(f64, @floatFromInt(self.normal_signal_trades))) * 100.0
        else
            0.0;

        // Average profit/loss for signal strength categories
        const strong_avg_profit = if (self.strong_signal_wins > 0)
            self.strong_signal_total_profit_pct / @as(f64, @floatFromInt(self.strong_signal_wins))
        else
            0.0;
        const strong_avg_loss = if (self.strong_signal_losses > 0)
            self.strong_signal_total_loss_pct / @as(f64, @floatFromInt(self.strong_signal_losses))
        else
            0.0;

        const normal_avg_profit = if (self.normal_signal_wins > 0)
            self.normal_signal_total_profit_pct / @as(f64, @floatFromInt(self.normal_signal_wins))
        else
            0.0;
        const normal_avg_loss = if (self.normal_signal_losses > 0)
            self.normal_signal_total_loss_pct / @as(f64, @floatFromInt(self.normal_signal_losses))
        else
            0.0;

        const open_positions = self.getOpenPositionsCount();

        std.log.info("=== PORTFOLIO PERFORMANCE ===", .{});
        std.log.info("Balance: ${d:.2} | Initial: ${d:.2} | Total Net P&L: ${d:.2} ({d:.1}%)", .{
            current_portfolio_value, self.initial_balance, total_pnl, total_return_pct,
        });
        std.log.info("Realized P&L: ${d:.2} | Unrealized P&L: ${d:.2}", .{
            self.total_realized_pnl, self.total_unrealized_pnl,
        });
        std.log.info("Closed Trades: {} | Win Rate: {d:.1}% | Loss Rate: {d:.1}%", .{
            closed_trades, overall_win_rate, overall_loss_rate,
        });
        std.log.info("Avg Profit: {d:.2}% | Avg Loss: {d:.2}% | Best: {d:.2}% | Worst: {d:.2}%", .{
            avg_profit_pct, avg_loss_pct, self.highest_profit_pct, self.lowest_loss_pct,
        });
        std.log.info("Strong Signals: {} trades | Win: {d:.1}% | Loss: {d:.1}% | Avg P: {d:.2}% | Avg L: {d:.2}% | Best: {d:.2}% | Worst: {d:.2}%", .{
            self.strong_signal_trades, strong_win_rate, strong_loss_rate, strong_avg_profit, strong_avg_loss, self.strong_signal_highest_profit_pct, self.strong_signal_lowest_loss_pct,
        });
        std.log.info("Normal Signals: {} trades | Win: {d:.1}% | Loss: {d:.1}% | Avg P: {d:.2}% | Avg L: {d:.2}% | Best: {d:.2}% | Worst: {d:.2}%", .{
            self.normal_signal_trades, normal_win_rate, normal_loss_rate, normal_avg_profit, normal_avg_loss, self.normal_signal_highest_profit_pct, self.normal_signal_lowest_loss_pct,
        });
        std.log.info("Open Positions: {} / {} | Available Balance: ${d:.2}", .{
            open_positions, self.max_positions, self.current_balance,
        });
        std.log.info("Position Sizing: Base: ${d:.2} | Min: ${d:.2} | Max: ${d:.2}", .{
            self.base_position_size_usdt, self.min_position_size_usdt, self.max_position_size_usdt,
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
        loss_rate: f64,
        avg_profit_pct: f64,
        avg_loss_pct: f64,
        highest_profit_pct: f64,
        lowest_loss_pct: f64,
        open_positions: usize,
        strong_signal_win_rate: f64,
        strong_signal_loss_rate: f64,
        strong_signal_avg_profit_pct: f64,
        strong_signal_avg_loss_pct: f64,
        strong_signal_highest_profit_pct: f64,
        strong_signal_lowest_loss_pct: f64,
        normal_signal_win_rate: f64,
        normal_signal_loss_rate: f64,
        normal_signal_avg_profit_pct: f64,
        normal_signal_avg_loss_pct: f64,
        normal_signal_highest_profit_pct: f64,
        normal_signal_lowest_loss_pct: f64,
        portfolio_value: f64,
    } {
        const total_pnl = self.total_realized_pnl + self.total_unrealized_pnl;
        const closed_trades = (self.total_trades - self.getOpenPositionsCount()) / 2;
        const win_rate = if (closed_trades > 0)
            (@as(f64, @floatFromInt(self.winning_trades)) / @as(f64, @floatFromInt(closed_trades))) * 100.0
        else
            0.0;
        const loss_rate = if (closed_trades > 0)
            (@as(f64, @floatFromInt(self.losing_trades)) / @as(f64, @floatFromInt(closed_trades))) * 100.0
        else
            0.0;

        const avg_profit_pct = if (self.winning_trades > 0)
            self.total_profit_pct / @as(f64, @floatFromInt(self.winning_trades))
        else
            0.0;
        const avg_loss_pct = if (self.losing_trades > 0)
            self.total_loss_pct / @as(f64, @floatFromInt(self.losing_trades))
        else
            0.0;

        const strong_win_rate = if (self.strong_signal_trades > 0)
            (@as(f64, @floatFromInt(self.strong_signal_wins)) / @as(f64, @floatFromInt(self.strong_signal_trades))) * 100.0
        else
            0.0;
        const strong_loss_rate = if (self.strong_signal_trades > 0)
            (@as(f64, @floatFromInt(self.strong_signal_losses)) / @as(f64, @floatFromInt(self.strong_signal_trades))) * 100.0
        else
            0.0;

        const normal_win_rate = if (self.normal_signal_trades > 0)
            (@as(f64, @floatFromInt(self.normal_signal_wins)) / @as(f64, @floatFromInt(self.normal_signal_trades))) * 100.0
        else
            0.0;
        const normal_loss_rate = if (self.normal_signal_trades > 0)
            (@as(f64, @floatFromInt(self.normal_signal_losses)) / @as(f64, @floatFromInt(self.normal_signal_trades))) * 100.0
        else
            0.0;

        // Average profit/loss for signal strength categories
        const strong_signal_avg_profit_pct = if (self.strong_signal_wins > 0)
            self.strong_signal_total_profit_pct / @as(f64, @floatFromInt(self.strong_signal_wins))
        else
            0.0;
        const strong_signal_avg_loss_pct = if (self.strong_signal_losses > 0)
            self.strong_signal_total_loss_pct / @as(f64, @floatFromInt(self.strong_signal_losses))
        else
            0.0;

        const normal_signal_avg_profit_pct = if (self.normal_signal_wins > 0)
            self.normal_signal_total_profit_pct / @as(f64, @floatFromInt(self.normal_signal_wins))
        else
            0.0;
        const normal_signal_avg_loss_pct = if (self.normal_signal_losses > 0)
            self.normal_signal_total_loss_pct / @as(f64, @floatFromInt(self.normal_signal_losses))
        else
            0.0;

        const portfolio_value = self.current_balance + self.total_unrealized_pnl;

        return .{
            .balance = self.current_balance,
            .total_pnl = total_pnl,
            .realized_pnl = self.total_realized_pnl,
            .unrealized_pnl = self.total_unrealized_pnl,
            .total_trades = closed_trades,
            .win_rate = win_rate,
            .loss_rate = loss_rate,
            .avg_profit_pct = avg_profit_pct,
            .avg_loss_pct = avg_loss_pct,
            .highest_profit_pct = self.highest_profit_pct,
            .lowest_loss_pct = self.lowest_loss_pct,
            .open_positions = self.getOpenPositionsCount(),
            .strong_signal_win_rate = strong_win_rate,
            .strong_signal_loss_rate = strong_loss_rate,
            .strong_signal_avg_profit_pct = strong_signal_avg_profit_pct,
            .strong_signal_avg_loss_pct = strong_signal_avg_loss_pct,
            .strong_signal_highest_profit_pct = self.strong_signal_highest_profit_pct,
            .strong_signal_lowest_loss_pct = self.strong_signal_lowest_loss_pct,
            .normal_signal_win_rate = normal_win_rate,
            .normal_signal_loss_rate = normal_loss_rate,
            .normal_signal_avg_profit_pct = normal_signal_avg_profit_pct,
            .normal_signal_avg_loss_pct = normal_signal_avg_loss_pct,
            .normal_signal_highest_profit_pct = self.normal_signal_highest_profit_pct,
            .normal_signal_lowest_loss_pct = self.normal_signal_lowest_loss_pct,
            .portfolio_value = portfolio_value,
        };
    }
};
