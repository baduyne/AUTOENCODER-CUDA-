#pragma once
#include <cstddef>

namespace dl {

/**
 * Conv2D forward (CPU, thuần float*)
 * input:   [N, Cin, H, W]
 * weight:  [Cout, Cin, K, K]
 * bias:    [Cout]
 * output:  [N, Cout, H_out, W_out]
 * - Caller cấp phát sẵn buffer output.
 * - Tính H_out, W_out = (H + 2*padding - K)/stride + 1
 */
void conv2d_forward_cpu(const float* input,
                        const float* weight,
                        const float* bias,
                        float* output,
                        int N, int Cin, int H, int W,
                        int Cout, int K, int stride, int padding);

/**
 * Conv2D backward (CPU, thuần float*)
 * Tính toàn bộ grad_input, grad_weight, grad_bias
 * - Caller cấp phát sẵn các buffer grad phù hợp.
 */
void conv2d_backward_cpu(const float* input,         // [N, Cin, H, W]
                         const float* weight,        // [Cout, Cin, K, K]
                         const float* grad_output,   // [N, Cout, H_out, W_out]
                         float* grad_input,          // [N, Cin, H, W]
                         float* grad_weight,         // [Cout, Cin, K, K]
                         float* grad_bias,           // [Cout]
                         int N, int Cin, int H, int W,
                         int Cout, int K, int stride, int padding);

} // namespace dl
