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

constexpr int CONV1_W_SIZE = C1 * 1 * K * K;
constexpr int CONV2_W_SIZE = C2 * C1 * K * K;
constexpr int FC_W_SIZE = FC_OUT * FC_IN;

constexpr int NUM_TRAIN_IMG = 8;

constexpr int MODE_INFER = 0;
constexpr int MODE_TRAIN = 1;

typedef ap_fixed<12, 4> w_t;
typedef ap_fixed<20, 8> acc_t;
typedef ap_fixed<12, 4> mem_t;
typedef ap_ufixed<12, 4> dw_t;

constexpr int STDP_TAU_PLUS = 4;
constexpr int STDP_TAU_MINUS = 4;
constexpr dw_t STDP_A_PLUS = dw_t(0.01);
constexpr dw_t STDP_A_MINUS = dw_t(0.012);
constexpr w_t W_MAX = w_t(3.0);
constexpr w_t W_MIN = w_t(-3.0);

constexpr dw_t LTP_LUT[T_STEPS] = {
    dw_t(0.01), dw_t(0.0075), dw_t(0.005625), dw_t(0.004219),
    dw_t(0.003164), dw_t(0.002373), dw_t(0.001780), dw_t(0.001335)};
constexpr dw_t LTD_LUT[T_STEPS] = {
    dw_t(0.012), dw_t(0.009), dw_t(0.00675), dw_t(0.005063),
    dw_t(0.003797), dw_t(0.002848), dw_t(0.002136), dw_t(0.001602)};

typedef ap_axiu<8, 0, 0, 0> axis_in_t;
typedef ap_axiu<16, 0, 0, 0> axis_out_t;

void snn_top(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream);

#endif
