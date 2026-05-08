#ifndef SNN_STDP_TOP_H_
#define SNN_STDP_TOP_H_

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

constexpr int IMG_H = 28;
constexpr int IMG_W = 28;
// Conservative synthesis profile: shrink channel counts first so the design is
// biased toward implementation closure instead of model capacity.
constexpr int C1 = 8;
constexpr int C2 = 8;
constexpr int K = 3;
constexpr int P1_H = 14;
constexpr int P1_W = 14;
constexpr int P2_H = 7;
constexpr int P2_W = 7;
constexpr int FC_IN = C2 * P2_H * P2_W;
constexpr int FC_OUT = 10;
constexpr int T_STEPS = 8;

constexpr int CONV1_W_SIZE = C1 * 1 * K * K;
constexpr int CONV2_W_SIZE = C2 * C1 * K * K;
constexpr int FC_W_SIZE = FC_OUT * FC_IN;
constexpr int TOTAL_WEIGHT_WORDS = CONV1_W_SIZE + CONV2_W_SIZE + FC_W_SIZE;

// MODE_TRAIN receives NUM_TRAIN_IMG support samples. Each sample is encoded as
// one label byte followed by one 28x28 image.
constexpr int NUM_TRAIN_IMG = 10;

constexpr int MODE_INFER = 0;
constexpr int MODE_TRAIN = 1;
constexpr int MODE_WEIGHTED_INFER = 2;
constexpr int MODE_WEIGHTED_TRAIN = 3;
constexpr int MODE_WEIGHTED_TRAIN_ONLY = 4;

typedef ap_uint<8> pix_t;
// Keep enough fractional precision so the small deterministic bootstrap
// weights used by the STDP demo do not quantize to zero after HLS lowering.
typedef ap_fixed<12, 4> w_t;
typedef ap_fixed<20, 8> acc_t;
typedef ap_fixed<16, 8> mem_t;
typedef ap_ufixed<12, 2> dw_t;
typedef ap_uint<4> ts_t;
typedef ap_uint<5> spike_cnt_t;

constexpr int STDP_TAU_PLUS = 4;
constexpr int STDP_TAU_MINUS = 4;
const dw_t STDP_A_PLUS = dw_t(0.01);
const dw_t STDP_A_MINUS = dw_t(0.012);
const w_t W_MAX = w_t(1.0);
const w_t W_MIN = w_t(-1.0);
const ts_t TS_NONE = ts_t((1 << 4) - 1);

typedef ap_axiu<8, 0, 0, 0> axis_in_t;
typedef ap_axiu<16, 0, 0, 0> axis_out_t;

void snn_top(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream);

#endif
