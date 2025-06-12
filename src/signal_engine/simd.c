#include <immintrin.h>
#include <stdbool.h>

typedef struct {
    bool should_generate_buy;
    bool should_generate_sell; 
    bool has_open_position;
    bool spread_valid;
    float signal_strength;
} TradingDecision;

void analyze_trading_signals_simd(
    float *rsi_values, 
    float *bid_percentages,
    float *ask_percentages, 
    float *spread_percentages,
    bool *has_positions,
    int len,
    TradingDecision *decisions
) {
    // calibrated thresholds for crypto HFT with 0.1% Binance fees
    const __m256 rsi_buy_threshold = _mm256_set1_ps(25.0f);
    const __m256 rsi_sell_threshold = _mm256_set1_ps(85.0f);
    const __m256 rsi_strong_buy = _mm256_set1_ps(15.0f);
    const __m256 rsi_strong_sell = _mm256_set1_ps(80.0f);
    
    // adjusted for crypto volatility and HFT
    const __m256 bid_strong_buy_threshold = _mm256_set1_ps(80.0f);
    const __m256 ask_strong_sell_threshold = _mm256_set1_ps(56.0f);
    const __m256 bid_buy_threshold = _mm256_set1_ps(85.0f);
    const __m256 ask_sell_threshold = _mm256_set1_ps(60.0f);
    
    // tighter spread for crypto HFT (need to overcome 0.2% round-trip fee)
    const __m256 spread_threshold = _mm256_set1_ps(0.0000f);
    const __m256 spread_optimal = _mm256_set1_ps(0.0000f);
    
    const __m256 rsi_valid_min = _mm256_set1_ps(0.0f);
    const __m256 rsi_valid_max = _mm256_set1_ps(100.0f);

    // for scalar loop
    const float rsi_buy_thresh = 25.0f;
    const float rsi_strong_buy_thresh = 15.0f;
    const float rsi_sell_thresh = 85.0f;
    const float rsi_strong_sell_thresh = 80.0f;
    const float bid_strong_thresh = 80.0f;
    const float bid_thresh = 85.0f;
    const float ask_strong_thresh = 56.0f;
    const float ask_thresh = 60.0f;
    const float spread_thresh = 0.0f;
    
    int i = 0;
    
    // 8 symbols at once
    for (; i <= len - 8; i += 8) {
        __m256 rsi_chunk = _mm256_loadu_ps(&rsi_values[i]);
        __m256 bid_chunk = _mm256_loadu_ps(&bid_percentages[i]);
        __m256 ask_chunk = _mm256_loadu_ps(&ask_percentages[i]);
        __m256 spread_chunk = _mm256_loadu_ps(&spread_percentages[i]);
        
        // bool array to float mask
        __m256 has_pos_mask = _mm256_setzero_ps();
        for (int j = 0; j < 8; j++) {
            if (has_positions[i + j]) {
                ((float*)&has_pos_mask)[j] = *(float*)&(int){0xFFFFFFFF};
            }
        }
        
        // RSI validity check
        __m256 rsi_valid = _mm256_and_ps(
            _mm256_cmp_ps(rsi_chunk, rsi_valid_min, _CMP_GE_OQ),
            _mm256_cmp_ps(rsi_chunk, rsi_valid_max, _CMP_LE_OQ)
        );
        
        // Spread validity checks (kept for output struct)
        __m256 spread_valid = _mm256_cmp_ps(spread_chunk, spread_threshold, _CMP_GE_OQ);
        
        // RSI conditions
        __m256 rsi_oversold = _mm256_cmp_ps(rsi_chunk, rsi_buy_threshold, _CMP_LE_OQ);
        __m256 rsi_very_oversold = _mm256_cmp_ps(rsi_chunk, rsi_strong_buy, _CMP_LE_OQ);
        
        // Orderbook conditions - if these are met, just buy/sell
        __m256 bid_strong = _mm256_cmp_ps(bid_chunk, bid_strong_buy_threshold, _CMP_GT_OQ);
        __m256 bid_threshold = _mm256_cmp_ps(bid_chunk, bid_buy_threshold, _CMP_GT_OQ);
        __m256 ask_strong = _mm256_cmp_ps(ask_chunk, ask_strong_sell_threshold, _CMP_GT_OQ);
        __m256 ask_threshold = _mm256_cmp_ps(ask_chunk, ask_sell_threshold, _CMP_GT_OQ);
        
        __m256 no_position = _mm256_xor_ps(has_pos_mask, _mm256_set1_ps(*(float*)&(int){0xFFFFFFFF}));
        
        // BUY CONDITIONS - Simplified (spread logic commented out)
        // Strong buy: Very oversold RSI + strong bid
        __m256 buy_strong = _mm256_and_ps(
            _mm256_and_ps(rsi_very_oversold, bid_strong),
            _mm256_and_ps(no_position, rsi_valid)
        );
        
        // Normal buy: Oversold RSI + bid threshold OR just strong bid signal
        __m256 buy_normal = _mm256_and_ps(
            _mm256_or_ps(
                _mm256_and_ps(rsi_oversold, bid_threshold),
                bid_strong
            ),
            // _mm256_and_ps(no_position, spread_valid)
            no_position
        );
        
        __m256 buy_condition = _mm256_or_ps(buy_strong, buy_normal);
        
        // RSI sell conditions
        __m256 rsi_overbought = _mm256_cmp_ps(rsi_chunk, rsi_sell_threshold, _CMP_GE_OQ);
        __m256 rsi_very_overbought = _mm256_cmp_ps(rsi_chunk, rsi_strong_sell, _CMP_GE_OQ);
        
        // RSI sell signals
        __m256 rsi_sell_strong = _mm256_and_ps(rsi_very_overbought, rsi_valid);
        __m256 rsi_sell_normal = _mm256_and_ps(rsi_overbought, rsi_valid);
        
        // SELL CONDITIONS - RSI OR orderbook, whichever happens first
        __m256 sell_condition = _mm256_and_ps(
            _mm256_or_ps(
                _mm256_or_ps(rsi_sell_strong, rsi_sell_normal),  // RSI conditions
                _mm256_or_ps(ask_strong, ask_threshold)          // Orderbook conditions
            ),
            // _mm256_and_ps(has_pos_mask, spread_valid)
            has_pos_mask
        );
        
        // Calculate signal strength
        __m256 signal_strength = _mm256_setzero_ps();
        __m256 strong_signal = _mm256_set1_ps(1.0f);
        __m256 normal_signal = _mm256_set1_ps(0.6f);
        
        // Strong buy signals
        signal_strength = _mm256_blendv_ps(signal_strength, strong_signal, buy_strong);
        // Normal buy signals (only if not already strong)
        signal_strength = _mm256_blendv_ps(signal_strength, normal_signal, 
            _mm256_andnot_ps(buy_strong, buy_normal));
        
        // Strong sell signals (RSI very overbought OR strong orderbook)
        __m256 sell_strong_signal = _mm256_or_ps(rsi_sell_strong, ask_strong);
        signal_strength = _mm256_blendv_ps(signal_strength, strong_signal, sell_strong_signal);
        
        // Normal sell signals (only if not already strong)
        __m256 sell_normal_signal = _mm256_andnot_ps(sell_strong_signal, 
            _mm256_or_ps(rsi_sell_normal, ask_threshold));
        signal_strength = _mm256_blendv_ps(signal_strength, normal_signal, sell_normal_signal);
        
        // Extract masks
        int buy_mask = _mm256_movemask_ps(buy_condition);
        int sell_mask = _mm256_movemask_ps(sell_condition);
        int spread_mask = _mm256_movemask_ps(spread_valid);
        
        // Store results
        for (int j = 0; j < 8; j++) {
            decisions[i + j].should_generate_buy = (buy_mask & (1 << j)) != 0;
            decisions[i + j].should_generate_sell = (sell_mask & (1 << j)) != 0;
            decisions[i + j].has_open_position = has_positions[i + j];
            decisions[i + j].spread_valid = (spread_mask & (1 << j)) != 0;
            decisions[i + j].signal_strength = ((float*)&signal_strength)[j];
        }
    }
    
    // scalar fallback for remaining elements
    for (; i < len; i++) {
        bool rsi_valid = rsi_values[i] >= 0.0f && rsi_values[i] <= 100.0f;
        bool spread_valid = spread_percentages[i] >= spread_thresh;
        
        float signal_strength = 0.0f;
        
        // buy logic - simplified (spread logic commented out)
        bool should_buy = false;
        // if (!has_positions[i] && spread_valid) {
        if (!has_positions[i]) {
            // Strong buy: Very oversold + strong bid
            if (rsi_valid && rsi_values[i] <= rsi_strong_buy_thresh && bid_percentages[i] > bid_strong_thresh) {
                should_buy = true;
                signal_strength = 1.0f;
            }
            // Normal buy: Oversold + bid threshold OR just strong bid
            else if ((rsi_valid && rsi_values[i] <= rsi_buy_thresh && bid_percentages[i] > bid_thresh) || 
                     bid_percentages[i] > bid_strong_thresh) {
                should_buy = true;
                signal_strength = 0.6f;
            }
        }
        
        // sell logic - RSI OR orderbook, whichever happens first
        bool should_sell = false;
        // if (has_positions[i] && spread_valid) {
        if (has_positions[i]) {
            // RSI sell conditions
            bool rsi_very_overbought = rsi_valid && rsi_values[i] >= rsi_strong_sell_thresh;
            bool rsi_overbought = rsi_valid && rsi_values[i] >= rsi_sell_thresh;
            
            // Orderbook sell conditions  
            bool ask_strong_signal = ask_percentages[i] > ask_strong_thresh;
            bool ask_threshold_signal = ask_percentages[i] > ask_thresh;
            
            // Sell if ANY condition is met
            if (rsi_very_overbought || rsi_overbought || ask_strong_signal || ask_threshold_signal) {
                should_sell = true;
                // Strong signal if very overbought RSI OR strong ask
                if (rsi_very_overbought || ask_strong_signal) {
                    signal_strength = 1.0f;
                } else {
                    signal_strength = 0.6f;
                }
            }
        }
        
        decisions[i].should_generate_buy = should_buy;
        decisions[i].should_generate_sell = should_sell;
        decisions[i].has_open_position = has_positions[i];
        decisions[i].spread_valid = spread_valid;
        decisions[i].signal_strength = signal_strength;
    }
}

// float calculate_position_size(float signal_strength, float base_size, float max_size) {
//     // Scale position size based on signal strength
//     // Strong signals (1.0) get full size, weaker signals get reduced size
//     float size = base_size * signal_strength;
//     return size > max_size ? max_size : size;
// }
