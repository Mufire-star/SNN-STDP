#ifndef TRUE_SNN_TOP_H_
#define TRUE_SNN_TOP_H_

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

constexpr int IMG_H = 28;
constexpr int IMG_W = 28;
constexpr int C1 = 16;
constexpr int C2 = 32;
constexpr int K = 3;
constexpr int P1_H = 14;
constexpr int P1_W = 14;
constexpr int P2_H = 7;
constexpr int P2_W = 7;
constexpr int FC_IN = C2 * P2_H * P2_W;
constexpr int FC_OUT = 10;
constexpr int T_STEPS = 8;

constexpr int PAR_K = 3;
constexpr int PAR_C1 = 4;
constexpr int PAR_FC_J = 7;
constexpr int PAR_FC_I = 2;

typedef ap_axiu<8, 0, 0, 0> axis_in_t;
typedef ap_axiu<16, 0, 0, 0> axis_out_t;

typedef ap_fixed<12, 4> w_t;
typedef ap_fixed<20, 8> acc_t;
typedef ap_fixed<20, 8> mem_t;

void snn_top(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream);

#endif
