#include <immintrin.h>
#include <stdbool.h>

typedef struct {
    bool should_generate_buy;
    bool should_generate_sell; 
    bool has_open_position;
    bool spread_valid;
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
    const __m256 rsi_buy_threshold = _mm256_set1_ps(20.0f);
    const __m256 rsi_sell_threshold = _mm256_set1_ps(80.0f);
    const __m256 bid_threshold = _mm256_set1_ps(70.0f);
    const __m256 ask_threshold = _mm256_set1_ps(50.0f);
    const __m256 spread_threshold = _mm256_set1_ps(0.2f);
    const __m256 rsi_valid_min = _mm256_set1_ps(0.0f);
    const __m256 rsi_valid_max = _mm256_set1_ps(100.0f);
    
    int i = 0;
    
    // 8 symbols
    for (; i <= len - 8; i += 8) {
        __m256 rsi_chunk = _mm256_loadu_ps(&rsi_values[i]);
        __m256 bid_chunk = _mm256_loadu_ps(&bid_percentages[i]);
        __m256 ask_chunk = _mm256_loadu_ps(&ask_percentages[i]);
        __m256 spread_chunk = _mm256_loadu_ps(&spread_percentages[i]);
        
        // convert bool array to float mask for SIMD operations
        __m256 has_pos_mask = _mm256_setzero_ps();
        for (int j = 0; j < 8; j++) {
            if (has_positions[i + j]) {
                // Set bit pattern for true
                ((float*)&has_pos_mask)[j] = *(float*)&(int){0xFFFFFFFF};
            }
        }
        
        // RSI validity check
        __m256 rsi_valid = _mm256_and_ps(
            _mm256_cmp_ps(rsi_chunk, rsi_valid_min, _CMP_GE_OQ),
            _mm256_cmp_ps(rsi_chunk, rsi_valid_max, _CMP_LE_OQ)
        );
        
        // Spread validity check
        __m256 spread_valid = _mm256_cmp_ps(spread_chunk, spread_threshold, _CMP_GE_OQ);
        
        // RSI conditions
        __m256 rsi_oversold = _mm256_cmp_ps(rsi_chunk, rsi_buy_threshold, _CMP_LE_OQ);
        __m256 rsi_overbought = _mm256_cmp_ps(rsi_chunk, rsi_sell_threshold, _CMP_GE_OQ);
        
        // Bid/Ask conditions
        __m256 bid_strong = _mm256_cmp_ps(bid_chunk, bid_threshold, _CMP_GT_OQ);
        __m256 ask_strong = _mm256_cmp_ps(ask_chunk, ask_threshold, _CMP_GT_OQ);
        
        // RSI oversold + strong bid + no position + valid spread + valid RSI
        __m256 no_position = _mm256_xor_ps(has_pos_mask, _mm256_set1_ps(*(float*)&(int){0xFFFFFFFF}));
        __m256 buy_condition = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_and_ps(rsi_oversold, bid_strong),
                _mm256_and_ps(no_position, spread_valid)
            ),
            rsi_valid
        );
        
        // (RSI overbought OR strong ask) + has position + valid spread
        __m256 sell_rsi_condition = _mm256_and_ps(rsi_overbought, rsi_valid);
        __m256 sell_ask_condition = ask_strong;
        __m256 sell_trigger = _mm256_or_ps(sell_rsi_condition, sell_ask_condition);
        __m256 sell_condition = _mm256_and_ps(
            _mm256_and_ps(sell_trigger, has_pos_mask),
            spread_valid
        );
        
        // extract masks
        int buy_mask = _mm256_movemask_ps(buy_condition);
        int sell_mask = _mm256_movemask_ps(sell_condition);
        int spread_mask = _mm256_movemask_ps(spread_valid);
        
        // store results
        for (int j = 0; j < 8; j++) {
            decisions[i + j].should_generate_buy = (buy_mask & (1 << j)) != 0;
            decisions[i + j].should_generate_sell = (sell_mask & (1 << j)) != 0;
            decisions[i + j].has_open_position = has_positions[i + j];
            decisions[i + j].spread_valid = (spread_mask & (1 << j)) != 0;
        }
    }
    
    // handle remaining elements (scalar fallback)
    for (; i < len; i++) {
        bool rsi_valid = rsi_values[i] >= 0.0f && rsi_values[i] <= 100.0f;
        bool spread_valid = spread_percentages[i] >= 0.2f;
        
        // Buy logic
        bool should_buy = rsi_valid && 
                         spread_valid &&
                         rsi_values[i] <= 20.0f && 
                         bid_percentages[i] > 70.0f && 
                         !has_positions[i];
        
        // Sell logic  
        bool should_sell = spread_valid &&
                          has_positions[i] && 
                          ((rsi_valid && rsi_values[i] >= 80.0f) || 
                           ask_percentages[i] > 50.0f);
        
        decisions[i].should_generate_buy = should_buy;
        decisions[i].should_generate_sell = should_sell;
        decisions[i].has_open_position = has_positions[i];
        decisions[i].spread_valid = spread_valid;
    }
}
