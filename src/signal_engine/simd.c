#include <immintrin.h>
#include <stdbool.h>

void analyze_rsi_simd(float *rsi_values, int len, bool *buy_signals, bool *sell_signals) {
    const __m256 buy_threshold = _mm256_set1_ps(20.0f);
    const __m256 sell_threshold = _mm256_set1_ps(80.0f);
    const __m256 valid_min = _mm256_set1_ps(0.0f);
    const __m256 valid_max = _mm256_set1_ps(100.0f);

    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 rsi_chunk = _mm256_loadu_ps(&rsi_values[i]);
        __m256 is_valid = _mm256_and_ps(
            _mm256_cmp_ps(rsi_chunk, valid_min, _CMP_GE_OQ),
            _mm256_cmp_ps(rsi_chunk, valid_max, _CMP_LE_OQ)
        );
        __m256 buy_condition = _mm256_and_ps(is_valid, _mm256_cmp_ps(rsi_chunk, buy_threshold, _CMP_LE_OQ));
        __m256 sell_condition = _mm256_and_ps(is_valid, _mm256_cmp_ps(rsi_chunk, sell_threshold, _CMP_GE_OQ));

        int buy_mask = _mm256_movemask_ps(buy_condition);
        int sell_mask = _mm256_movemask_ps(sell_condition);

        for (int j = 0; j < 8; j++) {
            buy_signals[i + j] = (buy_mask & (1 << j)) != 0;
            sell_signals[i + j] = (sell_mask & (1 << j)) != 0;
        }
    }
    for (; i < len; i++) {
        if (rsi_values[i] >= 0.0f && rsi_values[i] <= 100.0f) {
            buy_signals[i] = rsi_values[i] <= 20.0f;
            sell_signals[i] = rsi_values[i] >= 80.0f;
        } else {
            buy_signals[i] = false;
            sell_signals[i] = false;
        }
    }
}
