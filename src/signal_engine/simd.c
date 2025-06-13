#include <immintrin.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    float base_spread_threshold;
    float liquidity_multiplier;
    float volume_weighted_spread;
    float market_impact_penalty;
    bool is_liquid_enough;
} LiquidityAdjustedThreshold;

typedef struct {
    bool should_generate_buy;
    bool should_generate_sell; 
    bool has_open_position;
    bool spread_valid;
    bool liquidity_sufficient;
    float signal_strength;
    float adjusted_spread_threshold;
    float available_liquidity_ratio;
} TradingDecision;

LiquidityAdjustedThreshold calculate_liquidity_adjusted_threshold(
    float base_spread_threshold,
    float bid_volume,
    float ask_volume, 
    float intended_position_size_usdt,
    float best_bid,
    float best_ask
) {
    LiquidityAdjustedThreshold result = {0};
    
    // Calculate total available liquidity at best prices
    float total_liquidity_usdt = (bid_volume * best_bid) + (ask_volume * best_ask);
    
    // Liquidity ratio: how much of available liquidity our trade would consume
    float liquidity_consumption_ratio = intended_position_size_usdt / total_liquidity_usdt;
    
    // Base threshold adjustment based on liquidity consumption
    float liquidity_multiplier = 1.0f;
    if (liquidity_consumption_ratio > 0.5f) {
        // If we'd consume >50% of top-level liquidity, widen threshold significantly
        liquidity_multiplier = 1.0f + (liquidity_consumption_ratio * 2.0f);
    } else if (liquidity_consumption_ratio > 0.2f) {
        // If we'd consume 20-50%, moderate adjustment
        liquidity_multiplier = 1.0f + (liquidity_consumption_ratio * 0.5f);
    }
    
    // Market impact penalty for thin books
    float market_impact_penalty = 0.0f;
    if (total_liquidity_usdt < intended_position_size_usdt * 3.0f) {
        // If total liquidity < 3x our intended trade size, add penalty
        market_impact_penalty = base_spread_threshold * 0.5f;
    }
    
    // Volume-weighted spread calculation (simplified)
    float mid_price = (best_bid + best_ask) / 2.0f;
    float raw_spread = (best_ask - best_bid) / mid_price;
    
    result.base_spread_threshold = base_spread_threshold;
    result.liquidity_multiplier = liquidity_multiplier;
    result.volume_weighted_spread = raw_spread;
    result.market_impact_penalty = market_impact_penalty;
    result.is_liquid_enough = (liquidity_consumption_ratio < 0.3f) && (total_liquidity_usdt > intended_position_size_usdt);
    
    return result;
}

// Updated SIMD analysis function with liquidity consideration
void analyze_trading_signals_with_liquidity_simd(
    float *rsi_values,
    float *bid_percentages,
    float *ask_percentages,
    float *spread_percentages,
    float *bid_volumes,
    float *ask_volumes,
    float *best_bids,
    float *best_asks,
    float *position_sizes,
    bool *has_positions,
    int len,
    TradingDecision *decisions
) {
    const __m256 rsi_buy_threshold = _mm256_set1_ps(15.0f);
    const __m256 rsi_sell_threshold = _mm256_set1_ps(85.0f);
    const __m256 rsi_strong_buy = _mm256_set1_ps(10.0f);
    const __m256 rsi_strong_sell = _mm256_set1_ps(90.0f);
    const __m256 bid_strong_buy_threshold = _mm256_set1_ps(72.0f);
    const __m256 bid_buy_threshold = _mm256_set1_ps(60.0f);
    const __m256 ask_strong_sell_threshold = _mm256_set1_ps(46.0f);
    const __m256 ask_sell_threshold = _mm256_set1_ps(40.0f);

    const __m256 base_spread_threshold = _mm256_set1_ps(0.02f); // 0.02% base
    const __m256 max_spread_threshold = _mm256_set1_ps(0.005f);   // 0.5% max
    
    const __m256 min_liquidity_ratio = _mm256_set1_ps(3.0f);  // Min 3x our position size
    const __m256 max_consumption_ratio = _mm256_set1_ps(0.3f); // Max 30% of available liquidity
    
    int i = 0;
    
    // Process 8 symbols at once
    for (; i <= len - 8; i += 8) {
        __m256 rsi_chunk = _mm256_loadu_ps(&rsi_values[i]);
        __m256 bid_chunk = _mm256_loadu_ps(&bid_percentages[i]);
        __m256 ask_chunk = _mm256_loadu_ps(&ask_percentages[i]);
        __m256 spread_chunk = _mm256_loadu_ps(&spread_percentages[i]);
        
        // Load liquidity data
        __m256 bid_vol_chunk = _mm256_loadu_ps(&bid_volumes[i]);
        __m256 ask_vol_chunk = _mm256_loadu_ps(&ask_volumes[i]);
        __m256 best_bid_chunk = _mm256_loadu_ps(&best_bids[i]);
        __m256 best_ask_chunk = _mm256_loadu_ps(&best_asks[i]);
        __m256 position_size_chunk = _mm256_loadu_ps(&position_sizes[i]);
        
        // Load position status
        __m256 has_pos_mask = _mm256_setzero_ps();
        for (int j = 0; j < 8 && (i + j) < len; j++) {
            if (has_positions[i + j]) {
                ((float*)&has_pos_mask)[j] = *(float*)&(int){0xFFFFFFFF};
            }
        }
        
        // Calculate total available liquidity in USDT
        __m256 bid_liquidity_usdt = _mm256_mul_ps(bid_vol_chunk, best_bid_chunk);
        __m256 ask_liquidity_usdt = _mm256_mul_ps(ask_vol_chunk, best_ask_chunk);
        __m256 total_liquidity = _mm256_add_ps(bid_liquidity_usdt, ask_liquidity_usdt);
        
        // Calculate liquidity consumption ratio
        __m256 consumption_ratio = _mm256_div_ps(position_size_chunk, total_liquidity);
        
        // Liquidity-based threshold adjustment
        __m256 liquidity_multiplier = _mm256_set1_ps(1.0f);
        __m256 high_consumption = _mm256_cmp_ps(consumption_ratio, _mm256_set1_ps(0.5f), _CMP_GT_OQ);
        __m256 med_consumption = _mm256_cmp_ps(consumption_ratio, _mm256_set1_ps(0.2f), _CMP_GT_OQ);
        
        // Adjust multiplier based on consumption
        liquidity_multiplier = _mm256_blendv_ps(liquidity_multiplier, 
            _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(consumption_ratio, _mm256_set1_ps(2.0f))), 
            high_consumption);
        liquidity_multiplier = _mm256_blendv_ps(liquidity_multiplier,
            _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(consumption_ratio, _mm256_set1_ps(0.5f))),
            _mm256_andnot_ps(high_consumption, med_consumption));
        
        // Calculate adjusted spread threshold
        __m256 adjusted_threshold = _mm256_mul_ps(base_spread_threshold, liquidity_multiplier);
        adjusted_threshold = _mm256_min_ps(adjusted_threshold, max_spread_threshold);
        
        // Liquidity sufficiency check
        __m256 liquidity_sufficient = _mm256_and_ps(
            _mm256_cmp_ps(consumption_ratio, max_consumption_ratio, _CMP_LT_OQ),
            _mm256_cmp_ps(total_liquidity, _mm256_mul_ps(position_size_chunk, min_liquidity_ratio), _CMP_GT_OQ)
        );
        
        // Enhanced spread validity (original spread check + liquidity check)
        __m256 spread_valid = _mm256_and_ps(
            _mm256_cmp_ps(spread_chunk, adjusted_threshold, _CMP_LT_OQ),
            liquidity_sufficient
        );
        
        // RSI validity check
        __m256 rsi_valid = _mm256_and_ps(
            _mm256_cmp_ps(rsi_chunk, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
            _mm256_cmp_ps(rsi_chunk, _mm256_set1_ps(100.0f), _CMP_LE_OQ)
        );
        
        // RSI conditions
        __m256 rsi_oversold = _mm256_cmp_ps(rsi_chunk, rsi_buy_threshold, _CMP_LE_OQ);
        __m256 rsi_very_oversold = _mm256_cmp_ps(rsi_chunk, rsi_strong_buy, _CMP_LE_OQ);
        
        // Orderbook conditions - if these are met, just buy/sell
        __m256 bid_strong = _mm256_cmp_ps(bid_chunk, bid_strong_buy_threshold, _CMP_GT_OQ);
        __m256 bid_threshold = _mm256_cmp_ps(bid_chunk, bid_buy_threshold, _CMP_GT_OQ);
        __m256 ask_strong = _mm256_cmp_ps(ask_chunk, ask_strong_sell_threshold, _CMP_GT_OQ);
        __m256 ask_threshold = _mm256_cmp_ps(ask_chunk, ask_sell_threshold, _CMP_GT_OQ);
        
        __m256 no_position = _mm256_xor_ps(has_pos_mask, _mm256_set1_ps(*(float*)&(int){0xFFFFFFFF}));
        
        // BUY CONDITIONS - Simplified
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
            _mm256_and_ps(no_position, spread_valid)
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
            _mm256_and_ps(has_pos_mask, spread_valid)
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
        
        // Extract results and store with new liquidity fields
        int buy_mask = _mm256_movemask_ps(buy_condition);
        int sell_mask = _mm256_movemask_ps(sell_condition);
        int spread_mask = _mm256_movemask_ps(spread_valid);
        int liquidity_mask = _mm256_movemask_ps(liquidity_sufficient);
        
        for (int j = 0; j < 8 && (i + j) < len; j++) {
            decisions[i + j].should_generate_buy = (buy_mask & (1 << j)) != 0;
            decisions[i + j].should_generate_sell = (sell_mask & (1 << j)) != 0;
            decisions[i + j].spread_valid = (spread_mask & (1 << j)) != 0;
            decisions[i + j].liquidity_sufficient = (liquidity_mask & (1 << j)) != 0;
            decisions[i + j].adjusted_spread_threshold = ((float*)&adjusted_threshold)[j];
            decisions[i + j].available_liquidity_ratio = ((float*)&consumption_ratio)[j];
            decisions[i + j].signal_strength = ((float*)&signal_strength)[j];
            decisions[i + j].has_open_position = has_positions[i + j];
        }
    }
    
    // Scalar fallback for remaining elements
    for (; i < len; i++) {
        float rsi = rsi_values[i];
        float bid_pct = bid_percentages[i];
        float ask_pct = ask_percentages[i];
        float spread_pct = spread_percentages[i];
        
        LiquidityAdjustedThreshold liq_threshold = calculate_liquidity_adjusted_threshold(
            0.02f,  // 0.02% base threshold
            bid_volumes[i],
            ask_volumes[i],
            position_sizes[i],
            best_bids[i],
            best_asks[i]
        );
        
        float final_threshold = liq_threshold.base_spread_threshold * liq_threshold.liquidity_multiplier + 
                               liq_threshold.market_impact_penalty;
        final_threshold = fminf(final_threshold, 0.005f); // Cap at 0.5%
        
        bool spread_valid = (spread_pct < final_threshold) && liq_threshold.is_liquid_enough;
        bool rsi_valid = (rsi >= 0.0f && rsi <= 100.0f);
        bool has_position = has_positions[i];
        
        // Buy conditions
        bool buy_strong = !has_position && rsi_valid && rsi <= 10.0f && bid_pct > 72.0f;
        bool buy_normal = !has_position && spread_valid && 
                         ((rsi <= 15.0f && bid_pct > 60.0f) || bid_pct > 72.0f);
        
        // Sell conditions
        bool sell_strong = has_position && spread_valid && 
                          (rsi_valid && (rsi >= 90.0f || ask_pct > 46.0f));
        bool sell_normal = has_position && spread_valid && 
                          (rsi_valid && (rsi >= 85.0f || ask_pct > 40.0f));
        
        decisions[i].should_generate_buy = buy_strong || buy_normal;
        decisions[i].should_generate_sell = sell_strong || sell_normal;
        decisions[i].spread_valid = spread_valid;
        decisions[i].liquidity_sufficient = liq_threshold.is_liquid_enough;
        decisions[i].adjusted_spread_threshold = final_threshold;
        decisions[i].available_liquidity_ratio = position_sizes[i] / 
            ((bid_volumes[i] * best_bids[i]) + (ask_volumes[i] * best_asks[i]));
        decisions[i].signal_strength = buy_strong || sell_strong ? 1.0f : 
                                      (buy_normal || sell_normal ? 0.6f : 0.0f);
        decisions[i].has_open_position = has_position;
    }
}

float calculate_position_size(float signal_strength, float base_size, float max_size) {
    // Scale position size based on signal strength
    // Strong signals (1.0) get full size, weaker signals get reduced size
    float size = base_size * signal_strength;
    return size > max_size ? max_size : size;
}
