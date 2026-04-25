#include "snn_top.h"

static ap_uint<16> lfsr_step(ap_uint<16> state)
{
#pragma HLS INLINE
    ap_uint<1> b = state[0] ^ state[2] ^ state[3] ^ state[5];
    return (state >> 1) | (ap_uint<16>(b) << 15);
}

static inline int idx_conv1(int oc, int ki, int kj)
{
#pragma HLS INLINE
    return ((oc * K + ki) * K + kj);
}

static inline int idx_conv2(int oc, int ic, int ki, int kj)
{
#pragma HLS INLINE
    return (((oc * C1 + ic) * K + ki) * K + kj);
}

static inline int idx_fc(int o, int i)
{
#pragma HLS INLINE
    return o * FC_IN + i;
}

static inline w_t clamp_w(w_t w)
{
#pragma HLS INLINE
    if (w > W_MAX)
        return W_MAX;
    if (w < W_MIN)
        return W_MIN;
    return w;
}

// Use deterministic small positive initial weights so the project is
// self-contained and does not depend on an external supervised checkpoint.
static inline w_t init_conv1_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 11 + 7) % 11;
    switch (bucket)
    {
    case 0: return w_t(0.08);
    case 1: return w_t(0.09);
    case 2: return w_t(0.10);
    case 3: return w_t(0.11);
    case 4: return w_t(0.12);
    case 5: return w_t(0.13);
    case 6: return w_t(0.14);
    case 7: return w_t(0.15);
    case 8: return w_t(0.16);
    case 9: return w_t(0.17);
    default: return w_t(0.18);
    }
}

static inline w_t init_conv2_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 13 + 5) % 6;
    switch (bucket)
    {
    case 0: return w_t(0.030);
    case 1: return w_t(0.040);
    case 2: return w_t(0.050);
    case 3: return w_t(0.060);
    case 4: return w_t(0.070);
    default: return w_t(0.080);
    }
}

static inline w_t init_fc_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 17 + 3) % 5;
    switch (bucket)
    {
    case 0: return w_t(0.040);
    case 1: return w_t(0.055);
    case 2: return w_t(0.070);
    case 3: return w_t(0.085);
    default: return w_t(0.100);
    }
}

static inline w_t conv1_bias(int)
{
#pragma HLS INLINE
    return w_t(0.05);
}

static inline w_t conv2_bias(int)
{
#pragma HLS INLINE
    return w_t(0.06);
}

static inline w_t fc_bias(int)
{
#pragma HLS INLINE
    return w_t(0.12);
}

static inline dw_t stdp_ltp(int dt)
{
#pragma HLS INLINE
    switch (dt)
    {
    case 1: return dw_t(0.0075);
    case 2: return dw_t(0.005625);
    case 3: return dw_t(0.00421875);
    case 4: return dw_t(0.0031640625);
    case 5: return dw_t(0.002373046875);
    case 6: return dw_t(0.00177978515625);
    case 7: return dw_t(0.0013348388671875);
    default: return dw_t(0);
    }
}

static inline dw_t stdp_ltd(int dt)
{
#pragma HLS INLINE
    switch (dt)
    {
    case 1: return dw_t(0.009);
    case 2: return dw_t(0.00675);
    case 3: return dw_t(0.0050625);
    case 4: return dw_t(0.003796875);
    case 5: return dw_t(0.00284765625);
    case 6: return dw_t(0.0021357421875);
    case 7: return dw_t(0.001601806640625);
    default: return dw_t(0);
    }
}

void snn_top(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream)
{
#pragma HLS INTERFACE axis port = in_stream
#pragma HLS INTERFACE axis port = out_stream
#pragma HLS INTERFACE ap_ctrl_hs port = return

    w_t rw_conv1_w[CONV1_W_SIZE];
    w_t rw_conv2_w[CONV2_W_SIZE];
    w_t rw_fc_w[FC_W_SIZE];

#pragma HLS BIND_STORAGE variable = rw_conv1_w type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = rw_conv2_w type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = rw_fc_w type = ram_1p impl = bram

init_conv1:
    for (int i = 0; i < CONV1_W_SIZE; i++)
    {
#pragma HLS PIPELINE II = 1
        rw_conv1_w[i] = init_conv1_weight(i);
    }
init_conv2:
    for (int i = 0; i < CONV2_W_SIZE; i++)
    {
#pragma HLS PIPELINE II = 1
        rw_conv2_w[i] = init_conv2_weight(i);
    }
init_fc:
    for (int i = 0; i < FC_W_SIZE; i++)
    {
#pragma HLS PIPELINE II = 1
        rw_fc_w[i] = init_fc_weight(i);
    }

    axis_in_t mode_p = in_stream.read();
    ap_uint<1> mode = mode_p.data & ap_uint<1>(1);

    pix_t img[IMG_H][IMG_W];
    ap_uint<1> in_spk[IMG_H][IMG_W];
    mem_t mem1[C1][IMG_H][IMG_W];
    ap_uint<1> spk1[C1][IMG_H][IMG_W];
    ap_uint<1> p1[C1][P1_H][P1_W];
    mem_t mem2[C2][P1_H][P1_W];
    ap_uint<1> spk2[C2][P1_H][P1_W];
    ap_uint<1> p2[C2][P2_H][P2_W];
    mem_t mem3[FC_OUT];
    ap_uint<1> spk3[FC_OUT];
    spike_cnt_t spike_cnt[FC_OUT];
    ts_t last_pre_conv1[IMG_H][IMG_W];
    ts_t last_post_conv1[C1][IMG_H][IMG_W];
    ts_t last_pre_conv2[C1][P1_H][P1_W];
    ts_t last_post_conv2[C2][P1_H][P1_W];
    ts_t last_pre_fc[C2][P2_H][P2_W];
    ts_t last_post_fc[FC_OUT];

    // Keep intermediate spike maps in single-copy memories and reuse them over
    // time instead of letting HLS duplicate arrays for parallel reads.
#pragma HLS BIND_STORAGE variable = img type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = in_spk type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = mem1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = spk1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = p1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = mem2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = spk2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = p2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_conv1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_conv1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_conv2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_conv2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_fc type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_fc type = ram_1p impl = bram

    const mem_t v_th = mem_t(0.35);
    const mem_t alpha = mem_t(0.6);

    if (mode == MODE_TRAIN)
    {
    train_loop:
        for (int n = 0; n < NUM_TRAIN_IMG; n++)
        {

        reset_mem1:
            for (int oc = 0; oc < C1; oc++)
                for (int i = 0; i < IMG_H; i++)
                    for (int j = 0; j < IMG_W; j++)
                    {
#pragma HLS PIPELINE II = 1
                        mem1[oc][i][j] = mem_t(0);
                        spk1[oc][i][j] = ap_uint<1>(0);
                    }
        reset_p1:
            for (int oc = 0; oc < C1; oc++)
                for (int i = 0; i < P1_H; i++)
                    for (int j = 0; j < P1_W; j++)
                        p1[oc][i][j] = ap_uint<1>(0);
        reset_mem2:
            for (int oc = 0; oc < C2; oc++)
                for (int i = 0; i < P1_H; i++)
                    for (int j = 0; j < P1_W; j++)
                    {
#pragma HLS PIPELINE II = 1
                        mem2[oc][i][j] = mem_t(0);
                        spk2[oc][i][j] = ap_uint<1>(0);
                    }
        reset_p2:
            for (int oc = 0; oc < C2; oc++)
                for (int i = 0; i < P2_H; i++)
                    for (int j = 0; j < P2_W; j++)
                        p2[oc][i][j] = ap_uint<1>(0);
        reset_fc:
            for (int o = 0; o < FC_OUT; o++)
            {
                mem3[o] = mem_t(0);
                spk3[o] = ap_uint<1>(0);
                spike_cnt[o] = spike_cnt_t(0);
            }
        reset_ts_conv1_pre:
            for (int i = 0; i < IMG_H; i++)
                for (int j = 0; j < IMG_W; j++)
                    last_pre_conv1[i][j] = TS_NONE;
        reset_ts_conv1_post:
            for (int oc = 0; oc < C1; oc++)
                for (int i = 0; i < IMG_H; i++)
                    for (int j = 0; j < IMG_W; j++)
                        last_post_conv1[oc][i][j] = TS_NONE;
        reset_ts_conv2_pre:
            for (int ic = 0; ic < C1; ic++)
                for (int i = 0; i < P1_H; i++)
                    for (int j = 0; j < P1_W; j++)
                        last_pre_conv2[ic][i][j] = TS_NONE;
        reset_ts_conv2_post:
            for (int oc = 0; oc < C2; oc++)
                for (int i = 0; i < P1_H; i++)
                    for (int j = 0; j < P1_W; j++)
                        last_post_conv2[oc][i][j] = TS_NONE;
        reset_ts_fc_pre:
            for (int oc = 0; oc < C2; oc++)
                for (int i = 0; i < P2_H; i++)
                    for (int j = 0; j < P2_W; j++)
                        last_pre_fc[oc][i][j] = TS_NONE;
        reset_ts_fc_post:
            for (int o = 0; o < FC_OUT; o++)
                last_post_fc[o] = TS_NONE;

        load_train_img:
            for (int i = 0; i < IMG_H; i++)
            {
                for (int j = 0; j < IMG_W; j++)
                {
#pragma HLS PIPELINE II = 1
                    axis_in_t p = in_stream.read();
                    img[i][j] = pix_t(p.data);
                }
            }

            ap_uint<16> lfsr = 0xACE1;

        train_time_loop:
            for (int t = 0; t < T_STEPS; t++)
            {

            compute_in_spk_train:
                for (int i = 0; i < IMG_H; i++)
                {
                    for (int j = 0; j < IMG_W; j++)
                    {
#pragma HLS PIPELINE II = 1
                        lfsr = lfsr_step(lfsr);
                        pix_t rnd = pix_t(lfsr.range(7, 0));
                        in_spk[i][j] = (img[i][j] > rnd) ? ap_uint<1>(1) : ap_uint<1>(0);
                    }
                }

            update_pre_conv1:
                for (int i = 0; i < IMG_H; i++)
                {
                    for (int j = 0; j < IMG_W; j++)
                    {
#pragma HLS PIPELINE II = 1
                        if (in_spk[i][j])
                            last_pre_conv1[i][j] = ts_t(t);
                    }
                }

            conv1_lif_train:
                for (int oc = 0; oc < C1; oc++)
                {
                    for (int i = 0; i < IMG_H; i++)
                    {
                        for (int j = 0; j < IMG_W; j++)
                        {
                            acc_t sum = conv1_bias(oc);
                            for (int ki = 0; ki < K; ki++)
                            {
                                for (int kj = 0; kj < K; kj++)
                                {
                                    int ii = i + ki - 1;
                                    int jj = j + kj - 1;
                                    if (ii >= 0 && ii < IMG_H && jj >= 0 && jj < IMG_W)
                                    {
                                        if (in_spk[ii][jj])
                                        {
                                            sum += rw_conv1_w[idx_conv1(oc, ki, kj)];
                                        }
                                    }
                                }
                            }
                            mem_t m = mem1[oc][i][j] + alpha * (mem_t(sum) - mem1[oc][i][j]);
                            ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                            mem1[oc][i][j] = s ? mem_t(m - v_th) : mem_t(m);
                            spk1[oc][i][j] = s;
                            if (s)
                                last_post_conv1[oc][i][j] = ts_t(t);
                        }
                    }
                }

            stdp_conv1_ltp:
                for (int oc = 0; oc < C1; oc++)
                {
                    for (int i = 0; i < IMG_H; i++)
                    {
                        for (int j = 0; j < IMG_W; j++)
                        {
                            if (!spk1[oc][i][j])
                                continue;
                            for (int ki = 0; ki < K; ki++)
                            {
                                for (int kj = 0; kj < K; kj++)
                                {
                                    int ii = i + ki - 1;
                                    int jj = j + kj - 1;
                                    if (ii >= 0 && ii < IMG_H && jj >= 0 && jj < IMG_W)
                                    {
                                        int widx = idx_conv1(oc, ki, kj);
                                        if (in_spk[ii][jj])
                                        {
                                            rw_conv1_w[widx] = clamp_w(rw_conv1_w[widx] + w_t(STDP_A_PLUS));
                                        }
                                        else if (last_pre_conv1[ii][jj] != TS_NONE)
                                        {
                                            int dt = t - int(last_pre_conv1[ii][jj]);
                                            if (dt > 0 && dt < T_STEPS)
                                            {
                                                dw_t dw = stdp_ltp(dt);
                                                rw_conv1_w[widx] = clamp_w(rw_conv1_w[widx] + w_t(dw));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            stdp_conv1_ltd:
                for (int i = 0; i < IMG_H; i++)
                {
                    for (int j = 0; j < IMG_W; j++)
                    {
                        if (!in_spk[i][j])
                            continue;
                        for (int oc = 0; oc < C1; oc++)
                        {
                            if (last_post_conv1[oc][i][j] == TS_NONE)
                                continue;
                            int dt = t - int(last_post_conv1[oc][i][j]);
                            if (dt > 0 && dt < T_STEPS && !spk1[oc][i][j])
                            {
                                for (int ki = 0; ki < K; ki++)
                                {
                                    for (int kj = 0; kj < K; kj++)
                                    {
                                        int ii = i + ki - 1;
                                        int jj = j + kj - 1;
                                        if (ii >= 0 && ii < IMG_H && jj >= 0 && jj < IMG_W)
                                        {
                                            int widx = idx_conv1(oc, ki, kj);
                                            dw_t dw = stdp_ltd(dt);
                                            rw_conv1_w[widx] = clamp_w(rw_conv1_w[widx] - w_t(dw));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            pool1_train:
                for (int oc = 0; oc < C1; oc++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
                            ap_uint<1> v = 0;
                            for (int pi = 0; pi < 2; pi++)
                            {
                                for (int pj = 0; pj < 2; pj++)
                                {
                                    v |= spk1[oc][2 * i + pi][2 * j + pj];
                                }
                            }
                            p1[oc][i][j] = v;
                        }
                    }
                }

            update_pre_conv2:
                for (int ic = 0; ic < C1; ic++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
#pragma HLS PIPELINE II = 1
                            if (p1[ic][i][j])
                                last_pre_conv2[ic][i][j] = ts_t(t);
                        }
                    }
                }

            conv2_lif_train:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
                            acc_t sum = conv2_bias(oc);
                            for (int ic = 0; ic < C1; ic++)
                            {
                                for (int ki = 0; ki < K; ki++)
                                {
                                    for (int kj = 0; kj < K; kj++)
                                    {
                                        int ii = i + ki - 1;
                                        int jj = j + kj - 1;
                                        if (ii >= 0 && ii < P1_H && jj >= 0 && jj < P1_W)
                                        {
                                            if (p1[ic][ii][jj])
                                            {
                                                sum += rw_conv2_w[idx_conv2(oc, ic, ki, kj)];
                                            }
                                        }
                                    }
                                }
                            }
                            mem_t m = mem2[oc][i][j] + alpha * (mem_t(sum) - mem2[oc][i][j]);
                            ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                            mem2[oc][i][j] = s ? mem_t(m - v_th) : mem_t(m);
                            spk2[oc][i][j] = s;
                            if (s)
                                last_post_conv2[oc][i][j] = ts_t(t);
                        }
                    }
                }

            stdp_conv2_ltp:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
                            if (!spk2[oc][i][j])
                                continue;
                            for (int ic = 0; ic < C1; ic++)
                            {
                                for (int ki = 0; ki < K; ki++)
                                {
                                    for (int kj = 0; kj < K; kj++)
                                    {
                                        int ii = i + ki - 1;
                                        int jj = j + kj - 1;
                                        if (ii >= 0 && ii < P1_H && jj >= 0 && jj < P1_W)
                                        {
                                            int widx = idx_conv2(oc, ic, ki, kj);
                                            if (p1[ic][ii][jj])
                                            {
                                                rw_conv2_w[widx] = clamp_w(rw_conv2_w[widx] + w_t(STDP_A_PLUS));
                                            }
                                            else if (last_pre_conv2[ic][ii][jj] != TS_NONE)
                                            {
                                                int dt = t - int(last_pre_conv2[ic][ii][jj]);
                                                if (dt > 0 && dt < T_STEPS)
                                                {
                                                    dw_t dw = stdp_ltp(dt);
                                                    rw_conv2_w[widx] = clamp_w(rw_conv2_w[widx] + w_t(dw));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            stdp_conv2_ltd:
                for (int ic = 0; ic < C1; ic++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
                            if (!p1[ic][i][j])
                                continue;
                            for (int oc = 0; oc < C2; oc++)
                            {
                                for (int ki = 0; ki < K; ki++)
                                {
                                    for (int kj = 0; kj < K; kj++)
                                    {
                                        int ii = i + ki - 1;
                                        int jj = j + kj - 1;
                                        if (ii >= 0 && ii < P1_H && jj >= 0 && jj < P1_W)
                                        {
                                            if (last_post_conv2[oc][ii][jj] == TS_NONE)
                                                continue;
                                            int dt = t - int(last_post_conv2[oc][ii][jj]);
                                            if (dt > 0 && dt < T_STEPS && !spk2[oc][ii][jj])
                                            {
                                                int widx = idx_conv2(oc, ic, ki, kj);
                                                dw_t dw = stdp_ltd(dt);
                                                rw_conv2_w[widx] = clamp_w(rw_conv2_w[widx] - w_t(dw));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            pool2_train:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P2_H; i++)
                    {
                        for (int j = 0; j < P2_W; j++)
                        {
                            ap_uint<1> v = 0;
                            for (int pi = 0; pi < 2; pi++)
                            {
                                for (int pj = 0; pj < 2; pj++)
                                {
                                    v |= spk2[oc][2 * i + pi][2 * j + pj];
                                }
                            }
                            p2[oc][i][j] = v;
                        }
                    }
                }

            update_pre_fc:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P2_H; i++)
                    {
                        for (int j = 0; j < P2_W; j++)
                        {
#pragma HLS PIPELINE II = 1
                            if (p2[oc][i][j])
                                last_pre_fc[oc][i][j] = ts_t(t);
                        }
                    }
                }

            fc_lif_train:
                for (int o = 0; o < FC_OUT; o++)
                {
                    acc_t sum = fc_bias(o);
                    for (int oc = 0; oc < C2; oc++)
                    {
                        for (int i = 0; i < P2_H; i++)
                        {
                            for (int j = 0; j < P2_W; j++)
                            {
                                const int idx = ((oc * P2_H + i) * P2_W + j);
                                if (p2[oc][i][j])
                                {
                                    sum += rw_fc_w[idx_fc(o, idx)];
                                }
                            }
                        }
                    }
                    mem_t m = mem3[o] + alpha * (mem_t(sum) - mem3[o]);
                    ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                    mem3[o] = s ? mem_t(m - v_th) : mem_t(m);
                    spk3[o] = s;
                    spike_cnt[o] += s;
                    if (s)
                        last_post_fc[o] = ts_t(t);
                }

            stdp_fc_ltp:
                for (int o = 0; o < FC_OUT; o++)
                {
                    if (!spk3[o])
                        continue;
                    for (int oc = 0; oc < C2; oc++)
                    {
                        for (int i = 0; i < P2_H; i++)
                        {
                            for (int j = 0; j < P2_W; j++)
                            {
                                const int idx = ((oc * P2_H + i) * P2_W + j);
                                int widx = idx_fc(o, idx);
                                if (p2[oc][i][j])
                                {
                                    rw_fc_w[widx] = clamp_w(rw_fc_w[widx] + w_t(STDP_A_PLUS));
                                }
                                else if (last_pre_fc[oc][i][j] != TS_NONE)
                                {
                                    int dt = t - int(last_pre_fc[oc][i][j]);
                                    if (dt > 0 && dt < T_STEPS)
                                    {
                                        dw_t dw = stdp_ltp(dt);
                                        rw_fc_w[widx] = clamp_w(rw_fc_w[widx] + w_t(dw));
                                    }
                                }
                            }
                        }
                    }
                }
            stdp_fc_ltd:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P2_H; i++)
                    {
                        for (int j = 0; j < P2_W; j++)
                        {
                            if (!p2[oc][i][j])
                                continue;
                            for (int o = 0; o < FC_OUT; o++)
                            {
                                if (last_post_fc[o] == TS_NONE)
                                    continue;
                                int dt = t - int(last_post_fc[o]);
                                if (dt > 0 && dt < T_STEPS && !spk3[o])
                                {
                                    const int idx = ((oc * P2_H + i) * P2_W + j);
                                    int widx = idx_fc(o, idx);
                                    dw_t dw = stdp_ltd(dt);
                                    rw_fc_w[widx] = clamp_w(rw_fc_w[widx] - w_t(dw));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

infer_reset_mem1:
    for (int oc = 0; oc < C1; oc++)
        for (int i = 0; i < IMG_H; i++)
            for (int j = 0; j < IMG_W; j++)
            {
#pragma HLS PIPELINE II = 1
                mem1[oc][i][j] = mem_t(0);
                spk1[oc][i][j] = ap_uint<1>(0);
            }
infer_reset_p1:
    for (int oc = 0; oc < C1; oc++)
        for (int i = 0; i < P1_H; i++)
            for (int j = 0; j < P1_W; j++)
                p1[oc][i][j] = ap_uint<1>(0);
infer_reset_mem2:
    for (int oc = 0; oc < C2; oc++)
        for (int i = 0; i < P1_H; i++)
            for (int j = 0; j < P1_W; j++)
            {
#pragma HLS PIPELINE II = 1
                mem2[oc][i][j] = mem_t(0);
                spk2[oc][i][j] = ap_uint<1>(0);
            }
infer_reset_p2:
    for (int oc = 0; oc < C2; oc++)
        for (int i = 0; i < P2_H; i++)
            for (int j = 0; j < P2_W; j++)
                p2[oc][i][j] = ap_uint<1>(0);
infer_reset_fc:
    for (int o = 0; o < FC_OUT; o++)
    {
        mem3[o] = mem_t(0);
        spk3[o] = ap_uint<1>(0);
        spike_cnt[o] = spike_cnt_t(0);
    }

    load_infer_img:
    for (int i = 0; i < IMG_H; i++)
    {
        for (int j = 0; j < IMG_W; j++)
        {
#pragma HLS PIPELINE II = 1
            axis_in_t p = in_stream.read();
            img[i][j] = pix_t(p.data);
        }
    }

    {
        ap_uint<16> lfsr = 0xACE1;

    infer_time_loop:
        for (int t = 0; t < T_STEPS; t++)
        {

        compute_in_spk_infer:
            for (int i = 0; i < IMG_H; i++)
            {
                for (int j = 0; j < IMG_W; j++)
                {
#pragma HLS PIPELINE II = 1
                    lfsr = lfsr_step(lfsr);
                    pix_t rnd = pix_t(lfsr.range(7, 0));
                    in_spk[i][j] = (img[i][j] > rnd) ? ap_uint<1>(1) : ap_uint<1>(0);
                }
            }

        conv1_lif_infer:
            for (int oc = 0; oc < C1; oc++)
            {
                for (int i = 0; i < IMG_H; i++)
                {
                    for (int j = 0; j < IMG_W; j++)
                    {
                        acc_t sum = conv1_bias(oc);
                        for (int ki = 0; ki < K; ki++)
                        {
                            for (int kj = 0; kj < K; kj++)
                            {
                                int ii = i + ki - 1;
                                int jj = j + kj - 1;
                                if (ii >= 0 && ii < IMG_H && jj >= 0 && jj < IMG_W)
                                {
                                    if (in_spk[ii][jj])
                                    {
                                        sum += rw_conv1_w[idx_conv1(oc, ki, kj)];
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

        pool1_infer:
            for (int oc = 0; oc < C1; oc++)
            {
                for (int i = 0; i < P1_H; i++)
                {
                    for (int j = 0; j < P1_W; j++)
                    {
                        ap_uint<1> v = 0;
                        for (int pi = 0; pi < 2; pi++)
                        {
                            for (int pj = 0; pj < 2; pj++)
                            {
                                v |= spk1[oc][2 * i + pi][2 * j + pj];
                            }
                        }
                        p1[oc][i][j] = v;
                    }
                }
            }

        conv2_lif_infer:
            for (int oc = 0; oc < C2; oc++)
            {
                for (int i = 0; i < P1_H; i++)
                {
                    for (int j = 0; j < P1_W; j++)
                    {
                        acc_t sum = conv2_bias(oc);
                        for (int ic = 0; ic < C1; ic++)
                        {
                            for (int ki = 0; ki < K; ki++)
                            {
                                for (int kj = 0; kj < K; kj++)
                                {
                                    int ii = i + ki - 1;
                                    int jj = j + kj - 1;
                                    if (ii >= 0 && ii < P1_H && jj >= 0 && jj < P1_W)
                                    {
                                        if (p1[ic][ii][jj])
                                        {
                                            sum += rw_conv2_w[idx_conv2(oc, ic, ki, kj)];
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

        pool2_infer:
            for (int oc = 0; oc < C2; oc++)
            {
                for (int i = 0; i < P2_H; i++)
                {
                    for (int j = 0; j < P2_W; j++)
                    {
                        ap_uint<1> v = 0;
                        for (int pi = 0; pi < 2; pi++)
                        {
                            for (int pj = 0; pj < 2; pj++)
                            {
                                v |= spk2[oc][2 * i + pi][2 * j + pj];
                            }
                        }
                        p2[oc][i][j] = v;
                    }
                }
            }

        fc_lif_infer:
            for (int o = 0; o < FC_OUT; o++)
            {
                acc_t sum = fc_bias(o);
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P2_H; i++)
                    {
                        for (int j = 0; j < P2_W; j++)
                        {
                            const int idx = ((oc * P2_H + i) * P2_W + j);
                            if (p2[oc][i][j])
                            {
                                sum += rw_fc_w[idx_fc(o, idx)];
                            }
                        }
                    }
                }
                mem_t m = mem3[o] + alpha * (mem_t(sum) - mem3[o]);
                ap_uint<1> s = (m >= v_th) ? ap_uint<1>(1) : ap_uint<1>(0);
                mem3[o] = s ? mem_t(m - v_th) : mem_t(m);
                spk3[o] = s;
                spike_cnt[o] += s;
            }
        }
    }

output:
    for (int o = 0; o < FC_OUT; o++)
    {
#pragma HLS PIPELINE II = 1
        axis_out_t outp;
        ap_uint<16> q = (ap_uint<16>(spike_cnt[o]) * 256) / T_STEPS;
        outp.data = q;
        outp.keep = -1;
        outp.strb = -1;
        outp.last = (o == FC_OUT - 1) ? 1 : 0;
        out_stream.write(outp);
    }
}
