const std = @import("std");

pub const Error = error{
    OutOfMemory,
    InvalidSymbol,
    InvalidOrder,
    OrderNotFound,
    OrderBookNotFound,
    OrderBookFull,
    OrderBookEmpty,
    InvalidTrade,
    TradeNotFound,
    TradeBookNotFound,
    TradeBookFull,
    TradeBookEmpty,
    InvalidMarketData,
    MarketDataNotFound,
    MarketDataFull,
    MarketDataEmpty,
    NetworkError,
    BadResponse,
    Timeout,
    AuthenticationError,
    RateLimitExceeded,

    InvalidBidFormat,
    InvalidAskFormat,
    DepthRequestFailed,
    ExchangeInfoRequestFailed,
    InvalidDepthEvent,
};

pub const DepthError = error{
    DepthRequestFailed,
    OutOfMemory,
} || std.json.Error || std.http.Client.RequestError || std.Uri.ParseError;
