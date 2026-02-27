#include "snn_top.h"
#include "weights/weights_generated.h"

static ap_uint<16> lfsr_step(ap_uint<16> state) {
#pragma HLS INLINE
    ap_uint<1> b = state[0] ^ state[2] ^ state[3] ^ state[5];
    return (state >> 1) | (ap_uint<16>(b) << 15);
}

static inline int idx_conv1(int oc, int ki, int kj) {
#pragma HLS INLINE
    return ((oc * K + ki) * K + kj);
}

static inline int idx_conv2(int oc, int ic, int ki, int kj) {
#pragma HLS INLINE
    return (((oc * C1 + ic) * K + ki) * K + kj);
}

static inline int idx_fc(int o, int i) {
#pragma HLS INLINE
    return o * FC_IN + i;
}

void snn_top(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_hs port=return

#pragma HLS BIND_STORAGE variable=conv1_w type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_w type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=fc_w type=rom_1p impl=bram

    w_t img[IMG_H][IMG_W];

    mem_t mem1[C1][IMG_H][IMG_W] = {0};
    ap_uint<1> spk1[C1][IMG_H][IMG_W] = {0};
    ap_uint<1> p1[C1][P1_H][P1_W] = {0};

    mem_t mem2[C2][P1_H][P1_W] = {0};
    ap_uint<1> spk2[C2][P1_H][P1_W] = {0};
    ap_uint<1> p2[C2][P2_H][P2_W] = {0};

    mem_t mem3[FC_OUT] = {0};
    ap_uint<8> spike_cnt[FC_OUT] = {0};

    const mem_t v_th = mem_t(0.5);
    const mem_t alpha = mem_t(0.5); // tau=2.0 => alpha=1/tau

    // Load one image frame (0..255) and normalize to [0,1]
    for (int i = 0; i < IMG_H; i++) {
        for (int j = 0; j < IMG_W; j++) {
#pragma HLS PIPELINE II=1
            axis_in_t p = in_stream.read();
            img[i][j] = w_t(float(p.data) / 255.0f);
        }
    }

    ap_uint<16> lfsr = 0xACE1;

    for (int t = 0; t < T_STEPS; t++) {
        // Conv1(folded BN) + LIF
        for (int oc = 0; oc < C1; oc++) {
            for (int i = 0; i < IMG_H; i++) {
                for (int j = 0; j < IMG_W; j++) {
                    acc_t sum = conv1_b[oc];
                    for (int ki = 0; ki < K; ki++) {
#pragma HLS UNROLL factor=PAR_K
                        for (int kj = 0; kj < K; kj++) {
#pragma HLS UNROLL factor=PAR_K
                            int ii = i + ki - 1;
                            int jj = j + kj - 1;
                            if (ii >= 0 && ii < IMG_H && jj >= 0 && jj < IMG_W) {
                                lfsr = lfsr_step(lfsr);
                                w_t rnd = w_t(float(ap_uint<8>(lfsr.range(7, 0))) / 255.0f);
                                w_t in_spike = (img[ii][jj] > rnd) ? w_t(1) : w_t(0);
                                if (in_spike == w_t(1)) {
                                    sum += conv1_w[idx_conv1(oc, ki, kj)];
                                }
                            }
                        }
                    }
                    mem_t m = mem1[oc][i][j] + alpha * (mem_t(sum) - mem1[oc][i][j]);
                    ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                    mem1[oc][i][j] = s ? mem_t(m - v_th) : mem_t(m);
                    spk1[oc][i][j] = s;
                }
            }
        }

        // Pool1 (2x2 max on spikes)
        for (int oc = 0; oc < C1; oc++) {
            for (int i = 0; i < P1_H; i++) {
                for (int j = 0; j < P1_W; j++) {
#pragma HLS PIPELINE II=1
                    ap_uint<1> v = 0;
                    v |= spk1[oc][2 * i][2 * j];
                    v |= spk1[oc][2 * i + 1][2 * j];
                    v |= spk1[oc][2 * i][2 * j + 1];
                    v |= spk1[oc][2 * i + 1][2 * j + 1];
                    p1[oc][i][j] = v;
                }
            }
        }

        // Conv2(folded BN) + LIF
        for (int oc = 0; oc < C2; oc++) {
            for (int i = 0; i < P1_H; i++) {
                for (int j = 0; j < P1_W; j++) {
#pragma HLS PIPELINE II=1
                    acc_t sum = conv2_b[oc];
                    for (int ic = 0; ic < C1; ic++) {
#pragma HLS UNROLL factor=8
                        for (int ki = 0; ki < K; ki++) {
#pragma HLS UNROLL factor=PAR_K
                            for (int kj = 0; kj < K; kj++) {
#pragma HLS UNROLL factor=PAR_K
                                int ii = i + ki - 1;
                                int jj = j + kj - 1;
                                if (ii >= 0 && ii < P1_H && jj >= 0 && jj < P1_W) {
                                    if (p1[ic][ii][jj]) {
                                        sum += conv2_w[idx_conv2(oc, ic, ki, kj)];
                                    }
                                }
                            }
                        }
                    }
                    mem_t m = mem2[oc][i][j] + alpha * (mem_t(sum) - mem2[oc][i][j]);
                    ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                    mem2[oc][i][j] = s ? mem_t(m - v_th) : mem_t(m);
                    spk2[oc][i][j] = s;
                }
            }
        }

        // Pool2 (2x2 max on spikes)
        for (int oc = 0; oc < C2; oc++) {
            for (int i = 0; i < P2_H; i++) {
                for (int j = 0; j < P2_W; j++) {
#pragma HLS PIPELINE II=1
                    ap_uint<1> v = 0;
                    v |= spk2[oc][2 * i][2 * j];
                    v |= spk2[oc][2 * i + 1][2 * j];
                    v |= spk2[oc][2 * i][2 * j + 1];
                    v |= spk2[oc][2 * i + 1][2 * j + 1];
                    p2[oc][i][j] = v;
                }
            }
        }

        // FC + output LIF accumulation
        for (int o = 0; o < FC_OUT; o++) {
            acc_t sum = fc_b[o];
            for (int oc = 0; oc < C2; oc++) {
                for (int i = 0; i < P2_H; i++) {
#pragma HLS UNROLL factor=PAR_FC_I
                    for (int j = 0; j < P2_W; j++) {
#pragma HLS UNROLL factor=PAR_FC_J
                        const int idx = ((oc * P2_H + i) * P2_W + j);
                        if (p2[oc][i][j]) {
                            sum += fc_w[idx_fc(o, idx)];
                        }
                    }
                }
            }
            mem_t m = mem3[o] + alpha * (mem_t(sum) - mem3[o]);
            ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
            mem3[o] = s ? mem_t(m - v_th) : mem_t(m);
            spike_cnt[o] += s;
        }
    }

    // Output 10 spike rates scaled by 256/T (Q8-like)
    for (int o = 0; o < FC_OUT; o++) {
#pragma HLS PIPELINE II=1
        axis_out_t outp;
        ap_uint<16> q = (ap_uint<16>(spike_cnt[o]) * 256) / T_STEPS;
        outp.data = q;
        outp.keep = -1;
        outp.strb = -1;
        outp.last = (o == FC_OUT - 1) ? 1 : 0;
        out_stream.write(outp);
    }
}
