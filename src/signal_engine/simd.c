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
    const __m256 rsi_buy_threshold = _mm256_set1_ps(25.0f);    // less extreme for HFT
    const __m256 rsi_sell_threshold = _mm256_set1_ps(75.0f);   // less extreme for HFT
    const __m256 rsi_strong_buy = _mm256_set1_ps(15.0f);       // very oversold
    const __m256 rsi_strong_sell = _mm256_set1_ps(85.0f);      // very overbought
    
    // adjusted for crypto volatility and HFT
    const __m256 bid_threshold = _mm256_set1_ps(60.0f);        // lower for more signals
    const __m256 ask_threshold = _mm256_set1_ps(40.0f);        // lower for more signals
    const __m256 bid_strong_threshold = _mm256_set1_ps(80.0f); // strong bid pressure
    const __m256 ask_strong_threshold = _mm256_set1_ps(60.0f); // strong ask pressure
    
    // Tighter spread for crypto HFT (need to overcome 0.2% round-trip fee)
    const __m256 spread_threshold = _mm256_set1_ps(0.25f);     // 0.25% minimum spread
    const __m256 spread_optimal = _mm256_set1_ps(0.35f);       // optimal spread for larger size
    
    const __m256 rsi_valid_min = _mm256_set1_ps(0.0f);
    const __m256 rsi_valid_max = _mm256_set1_ps(100.0f);
    
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
        
        // Spread validity checks
        __m256 spread_valid = _mm256_cmp_ps(spread_chunk, spread_threshold, _CMP_GE_OQ);
        __m256 spread_optimal_check = _mm256_cmp_ps(spread_chunk, spread_optimal, _CMP_GE_OQ);
        
        // RSI conditions - multiple levels
        __m256 rsi_oversold = _mm256_cmp_ps(rsi_chunk, rsi_buy_threshold, _CMP_LE_OQ);
        __m256 rsi_very_oversold = _mm256_cmp_ps(rsi_chunk, rsi_strong_buy, _CMP_LE_OQ);
        __m256 rsi_overbought = _mm256_cmp_ps(rsi_chunk, rsi_sell_threshold, _CMP_GE_OQ);
        __m256 rsi_very_overbought = _mm256_cmp_ps(rsi_chunk, rsi_strong_sell, _CMP_GE_OQ);
        
        // Bid/Ask conditions - multiple levels
        __m256 bid_good = _mm256_cmp_ps(bid_chunk, bid_threshold, _CMP_GT_OQ);
        __m256 bid_strong = _mm256_cmp_ps(bid_chunk, bid_strong_threshold, _CMP_GT_OQ);
        __m256 ask_good = _mm256_cmp_ps(ask_chunk, ask_threshold, _CMP_GT_OQ);
        __m256 ask_strong = _mm256_cmp_ps(ask_chunk, ask_strong_threshold, _CMP_GT_OQ);
        
        __m256 no_position = _mm256_xor_ps(has_pos_mask, _mm256_set1_ps(*(float*)&(int){0xFFFFFFFF}));
        
        // BUY CONDITIONS - Multiple tiers
        // Tier 1: Strong RSI + Strong bid + optimal spread
        __m256 buy_tier1 = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_and_ps(rsi_very_oversold, bid_strong),
                _mm256_and_ps(no_position, spread_optimal_check)
            ),
            rsi_valid
        );
        
        // Tier 2: Normal RSI + good bid + valid spread
        __m256 buy_tier2 = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_and_ps(rsi_oversold, bid_good),
                _mm256_and_ps(no_position, spread_valid)
            ),
            rsi_valid
        );
        
        __m256 buy_condition = _mm256_or_ps(buy_tier1, buy_tier2);
        
        // SELL CONDITIONS - Multiple tiers
        // Tier 1: Very overbought RSI OR strong ask pressure
        __m256 sell_rsi_strong = _mm256_and_ps(rsi_very_overbought, rsi_valid);
        __m256 sell_ask_strong = ask_strong;
        __m256 sell_tier1 = _mm256_or_ps(sell_rsi_strong, sell_ask_strong);
        
        // Tier 2: Normal overbought OR good ask pressure
        __m256 sell_rsi_normal = _mm256_and_ps(rsi_overbought, rsi_valid);
        __m256 sell_ask_normal = ask_good;
        __m256 sell_tier2 = _mm256_or_ps(sell_rsi_normal, sell_ask_normal);
        
        // Apply position and spread filters
        __m256 sell_condition_tier1 = _mm256_and_ps(
            _mm256_and_ps(sell_tier1, has_pos_mask),
            spread_valid
        );
        
        __m256 sell_condition_tier2 = _mm256_and_ps(
            _mm256_and_ps(sell_tier2, has_pos_mask),
            spread_optimal_check  // Require better spread for tier 2
        );
        
        __m256 sell_condition = _mm256_or_ps(sell_condition_tier1, sell_condition_tier2);
        
        // Calculate signal strength for position sizing
        __m256 signal_strength = _mm256_setzero_ps();
        // Strong signals get higher strength
        __m256 strong_buy_strength = _mm256_set1_ps(1.0f);
        __m256 normal_buy_strength = _mm256_set1_ps(0.6f);
        __m256 strong_sell_strength = _mm256_set1_ps(1.0f);
        __m256 normal_sell_strength = _mm256_set1_ps(0.6f);
        
        signal_strength = _mm256_blendv_ps(signal_strength, strong_buy_strength, buy_tier1);
        signal_strength = _mm256_blendv_ps(signal_strength, normal_buy_strength, 
            _mm256_andnot_ps(buy_tier1, buy_tier2));
        signal_strength = _mm256_blendv_ps(signal_strength, strong_sell_strength, sell_condition_tier1);
        signal_strength = _mm256_blendv_ps(signal_strength, normal_sell_strength, 
            _mm256_andnot_ps(sell_condition_tier1, sell_condition_tier2));
        
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
        bool spread_valid = spread_percentages[i] >= 0.25f;
        bool spread_optimal = spread_percentages[i] >= 0.35f;
        
        float signal_strength = 0.0f;
        
        // buy logic - tiered approach
        bool should_buy = false;
        if (rsi_valid && !has_positions[i]) {
            // Tier 1: Very oversold + strong bid + optimal spread
            if (rsi_values[i] <= 15.0f && bid_percentages[i] > 80.0f && spread_optimal) {
                should_buy = true;
                signal_strength = 1.0f;
            }
            // Tier 2: Oversold + good bid + valid spread  
            else if (rsi_values[i] <= 25.0f && bid_percentages[i] > 60.0f && spread_valid) {
                should_buy = true;
                signal_strength = 0.6f;
            }
        }
        
        // sell logic - tiered approach
        bool should_sell = false;
        if (spread_valid && has_positions[i]) {
            // Tier 1: Very overbought OR strong ask
            if ((rsi_valid && rsi_values[i] >= 85.0f) || ask_percentages[i] > 60.0f) {
                should_sell = true;
                signal_strength = 1.0f;
            }
            // Tier 2: Overbought OR good ask (need optimal spread)
            else if (spread_optimal && 
                    ((rsi_valid && rsi_values[i] >= 75.0f) || ask_percentages[i] > 40.0f)) {
                should_sell = true;
                signal_strength = 0.6f;
            }
        }
        
        decisions[i].should_generate_buy = should_buy;
        decisions[i].should_generate_sell = should_sell;
        decisions[i].has_open_position = has_positions[i];
        decisions[i].spread_valid = spread_valid;
        decisions[i].signal_strength = signal_strength;
    }
}

float calculate_position_size(float signal_strength, float base_size, float max_size) {
    // Scale position size based on signal strength
    // Strong signals (1.0) get full size, weaker signals get reduced size
    float size = base_size * signal_strength;
    return size > max_size ? max_size : size;
}
