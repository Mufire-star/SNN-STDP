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

static inline w_t clamp_conv_w(w_t w)
{
#pragma HLS INLINE
    if (w > CONV_W_MAX)
        return CONV_W_MAX;
    if (w < CONV_W_MIN)
        return CONV_W_MIN;
    return w;
}

static inline w_t conv_ltp(w_t w, dw_t dw)
{
#pragma HLS INLINE
    ap_fixed<24, 8> span = ap_fixed<24, 8>(CONV_W_MAX) - ap_fixed<24, 8>(w);
    ap_fixed<24, 8> delta = ap_fixed<24, 8>(dw) * span;
    return clamp_conv_w(w + w_t(delta));
}

static inline w_t conv_ltd(w_t w, dw_t dw)
{
#pragma HLS INLINE
    ap_fixed<24, 8> span = ap_fixed<24, 8>(w) - ap_fixed<24, 8>(CONV_W_MIN);
    ap_fixed<24, 8> delta = ap_fixed<24, 8>(dw) * span;
    return clamp_conv_w(w - w_t(delta));
}

static inline w_t conv_homeo(w_t w, w_t target)
{
#pragma HLS INLINE
    ap_fixed<24, 8> diff = ap_fixed<24, 8>(target) - ap_fixed<24, 8>(w);
    ap_fixed<24, 8> delta = ap_fixed<24, 8>(CONV_HOMEO) * diff;
    return clamp_conv_w(w + w_t(delta));
}

static inline w_t conv_home_target(w_t init_w)
{
#pragma HLS INLINE
    return clamp_conv_w(init_w + CONV_HOME_OFFSET);
}

// Use deterministic bootstrap weights so the project is self-contained and
// does not depend on an external supervised checkpoint. The convolutional
// layers still adapt with unsupervised STDP during MODE_TRAIN; the FC layer
// starts mildly inhibitory so label-gated STDP can bind support samples to the
// matching output neurons.
static inline w_t init_conv1_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 5 + 7) % 11;
    switch (bucket)
    {
    case 0: return w_t(0.035);
    case 1: return w_t(0.039);
    case 2: return w_t(0.043);
    case 3: return w_t(0.047);
    case 4: return w_t(0.051);
    case 5: return w_t(0.055);
    case 6: return w_t(0.059);
    case 7: return w_t(0.063);
    case 8: return w_t(0.067);
    case 9: return w_t(0.071);
    default: return w_t(0.075);
    }
}

static inline w_t init_conv2_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 13 + 5) % 6;
    switch (bucket)
    {
    case 0: return w_t(0.020);
    case 1: return w_t(0.027);
    case 2: return w_t(0.034);
    case 3: return w_t(0.041);
    case 4: return w_t(0.048);
    default: return w_t(0.055);
    }
}

static inline w_t init_fc_weight(int idx)
{
#pragma HLS INLINE
    const int bucket = (idx * 7 + 3) % 5;
    switch (bucket)
    {
    case 0: return w_t(-0.050);
    case 1: return w_t(-0.046);
    case 2: return w_t(-0.042);
    case 3: return w_t(-0.038);
    default: return w_t(-0.034);
    }
}

static inline w_t conv1_bias(int)
{
#pragma HLS INLINE
    return w_t(0.0);
}

static inline w_t conv2_bias(int)
{
#pragma HLS INLINE
    return w_t(0.0);
}

static inline w_t fc_bias(int)
{
#pragma HLS INLINE
    return w_t(0.0);
}

static inline w_t fc_label_ltp(int dt)
{
#pragma HLS INLINE
    switch (dt)
    {
    case 0: return w_t(0.0234375);
    case 1: return w_t(0.015625);
    case 2: return w_t(0.01171875);
    case 3: return w_t(0.0078125);
    default: return w_t(0.00390625);
    }
}

static inline dw_t stdp_ltp(int dt)
{
#pragma HLS INLINE
    switch (dt)
    {
    case 1: return dw_t(0.001);
    case 2: return dw_t(0.00075);
    case 3: return dw_t(0.0005);
    default: return dw_t(0);
    }
}

static inline dw_t stdp_ltd(int dt)
{
#pragma HLS INLINE
    switch (dt)
    {
    case 1: return dw_t(0.0005);
    case 2: return dw_t(0.0005);
    case 3: return dw_t(0.00025);
    case 4: return dw_t(0.00025);
    case 5: return dw_t(0.00025);
    case 6: return dw_t(0.00025);
    default: return dw_t(0);
    }
}

static inline ap_int<12> score_bias(int o)
{
#pragma HLS INLINE
    switch (o)
    {
    case 0: return ap_int<12>(65);
    case 1: return ap_int<12>(8);
    case 2: return ap_int<12>(18);
    case 3: return ap_int<12>(94);
    case 4: return ap_int<12>(2);
    case 5: return ap_int<12>(0);
    case 6: return ap_int<12>(21);
    case 7: return ap_int<12>(-16);
    case 8: return ap_int<12>(-8);
    case 9: return ap_int<12>(-13);
    default: return ap_int<12>(0);
    }
}

static inline ap_int<8> read_s8(hls::stream<axis_in_t> &in_stream)
{
#pragma HLS INLINE
    axis_in_t p = in_stream.read();
    ap_uint<8> raw = p.data;
    return ap_int<8>(raw);
}

static inline w_t decode_weight_word(ap_int<8> raw)
{
#pragma HLS INLINE
    ap_fixed<24, 16> scaled = ap_fixed<24, 16>(raw) * ap_fixed<24, 16>(0.0078125);
    return w_t(scaled);
}

static inline ap_uint<16> encode_weight_word(w_t w)
{
#pragma HLS INLINE
    ap_fixed<24, 16> scaled = ap_fixed<24, 16>(w) * ap_fixed<24, 16>(128.0);
    ap_int<16> raw;
    if (scaled > ap_fixed<24, 16>(127.0))
        raw = ap_int<16>(127);
    else if (scaled < ap_fixed<24, 16>(-128.0))
        raw = ap_int<16>(-128);
    else if (scaled >= ap_fixed<24, 16>(0.0))
        raw = ap_int<16>(scaled + ap_fixed<24, 16>(0.5));
    else
        raw = ap_int<16>(scaled - ap_fixed<24, 16>(0.5));
    ap_int<8> q = ap_int<8>(raw);
    return ap_uint<16>(ap_int<16>(q));
}

static inline void write_u16(hls::stream<axis_out_t> &out_stream, ap_uint<16> data, bool last)
{
#pragma HLS INLINE
    axis_out_t outp;
    outp.data = data;
    outp.keep = -1;
    outp.strb = -1;
    outp.last = last ? 1 : 0;
    out_stream.write(outp);
}

static inline void diag_inc(ap_uint<16> &v)
{
#pragma HLS INLINE
    if (v != ap_uint<16>(0xFFFF))
        v++;
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

    axis_in_t mode_p = in_stream.read();
    ap_uint<8> mode = mode_p.data;
    const bool weighted_mode =
        (mode == MODE_WEIGHTED_INFER) ||
        (mode == MODE_WEIGHTED_TRAIN) ||
        (mode == MODE_WEIGHTED_TRAIN_ONLY);
    const bool train_mode =
        (mode == MODE_TRAIN) ||
        (mode == MODE_WEIGHTED_TRAIN) ||
        (mode == MODE_WEIGHTED_TRAIN_ONLY);
    const bool infer_mode =
        (mode == MODE_INFER) ||
        (mode == MODE_TRAIN) ||
        (mode == MODE_WEIGHTED_INFER) ||
        (mode == MODE_WEIGHTED_TRAIN);

    if (weighted_mode)
    {
    read_conv1:
        for (int i = 0; i < CONV1_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            rw_conv1_w[i] = decode_weight_word(read_s8(in_stream));
        }
    read_conv2:
        for (int i = 0; i < CONV2_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            rw_conv2_w[i] = decode_weight_word(read_s8(in_stream));
        }
    read_fc:
        for (int i = 0; i < FC_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            rw_fc_w[i] = decode_weight_word(read_s8(in_stream));
        }
    }
    else
    {
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
    }

    pix_t img[IMG_H][IMG_W];
    ap_uint<1> in_spk[IMG_H][IMG_W];
    mem_t mem1[C1][IMG_H][IMG_W];
    ap_uint<1> spk1[C1][IMG_H][IMG_W];
    ap_uint<1> p1[C1][P1_H][P1_W];
    mem_t mem2[C2][P1_H][P1_W];
    ap_uint<1> spk2[C2][P1_H][P1_W];
    ap_uint<1> p2[C2][P2_H][P2_W];
    mem_t mem3[FC_OUT];
    mem_t fc_next[FC_OUT];
    ap_uint<1> spk3[FC_OUT];
    spike_cnt_t spike_cnt[FC_OUT];
    ap_uint<1> fc_seen[FC_IN];
    ap_uint<4> fc_count[FC_IN];
    ts_t last_pre_conv1[IMG_H][IMG_W];
    ts_t last_post_conv1[C1][IMG_H][IMG_W];
    ts_t last_pre_conv2[C1][P1_H][P1_W];
    ts_t last_post_conv2[C2][P1_H][P1_W];
    ts_t last_pre_fc[C2][P2_H][P2_W];
    ts_t last_post_fc[FC_OUT];
    ap_uint<1> conv1_ltp_seen[CONV1_W_SIZE];
    ap_uint<1> conv1_ltd_seen[CONV1_W_SIZE];
    ap_uint<1> conv2_ltp_seen[CONV2_W_SIZE];
    ap_uint<1> conv2_ltd_seen[CONV2_W_SIZE];
    ap_fixed<24, 12> bp_feedback_c1[C1];
    ap_fixed<24, 12> bp_feedback_c2[C2];
    ap_uint<16> diag_correct_pre = 0;
    ap_uint<16> diag_wrong_pre = 0;
    ap_uint<16> diag_active_features = 0;
    ap_uint<16> diag_stdp_conv1 = 0;
    ap_uint<16> diag_stdp_conv2 = 0;
    ap_uint<16> diag_bp_conv1 = 0;
    ap_uint<16> diag_bp_conv2 = 0;
    ap_uint<16> diag_bp_fc_ltp = 0;
    ap_uint<16> diag_bp_fc_ltd = 0;

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
#pragma HLS BIND_STORAGE variable = fc_seen type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = fc_count type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_conv1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_conv1 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_conv2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_conv2 type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_pre_fc type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = last_post_fc type = ram_1p impl = bram

    const mem_t v_th = mem_t(0.35);
    const mem_t fc_v_th = mem_t(0.12);
    const mem_t alpha = mem_t(0.6);

    if (train_mode)
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
        reset_fc_seen_train:
            for (int i = 0; i < FC_IN; i++)
            {
#pragma HLS PIPELINE II = 1
                fc_seen[i] = ap_uint<1>(0);
                fc_count[i] = ap_uint<4>(0);
            }
        reset_ts_fc_post:
            for (int o = 0; o < FC_OUT; o++)
                last_post_fc[o] = TS_NONE;

            axis_in_t label_p = in_stream.read();
            ap_uint<4> train_label = label_p.data.range(3, 0);
            if (train_label >= FC_OUT)
                train_label = 0;

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

        reset_conv1_stdp_marks:
            for (int widx = 0; widx < CONV1_W_SIZE; widx++)
            {
#pragma HLS PIPELINE II = 1
                conv1_ltp_seen[widx] = ap_uint<1>(0);
                conv1_ltd_seen[widx] = ap_uint<1>(0);
            }
        reset_conv2_stdp_marks:
            for (int widx = 0; widx < CONV2_W_SIZE; widx++)
            {
#pragma HLS PIPELINE II = 1
                conv2_ltp_seen[widx] = ap_uint<1>(0);
                conv2_ltd_seen[widx] = ap_uint<1>(0);
            }

            {
                ap_uint<16> lfsr = ap_uint<16>(0xACE1u + ap_uint<16>(n * 97));

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
                        }
                    }
                }

            const bool enable_conv_stdp = true;
            if (enable_conv_stdp)
            {
            stdp_conv1:
                for (int oc = 0; oc < C1; oc++)
                {
                    for (int i = 0; i < IMG_H; i++)
                    {
                        for (int j = 0; j < IMG_W; j++)
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
                                        if (spk1[oc][i][j])
                                        {
                                            if (in_spk[ii][jj] && !conv1_ltp_seen[widx])
                                            {
                                                rw_conv1_w[widx] = conv_ltp(rw_conv1_w[widx], STDP_A_PLUS);
                                                conv1_ltp_seen[widx] = ap_uint<1>(1);
                                                diag_inc(diag_stdp_conv1);
                                            }
                                            else if (last_pre_conv1[ii][jj] != TS_NONE && !conv1_ltp_seen[widx])
                                            {
                                                int dt = t - int(last_pre_conv1[ii][jj]);
                                                if (dt > 0 && dt < T_STEPS)
                                                {
                                                    dw_t dw = stdp_ltp(dt);
                                                    rw_conv1_w[widx] = conv_ltp(rw_conv1_w[widx], dw);
                                                    conv1_ltp_seen[widx] = ap_uint<1>(1);
                                                    diag_inc(diag_stdp_conv1);
                                                }
                                            }
                                        }
                                        if (in_spk[ii][jj] && last_post_conv1[oc][i][j] != TS_NONE && !conv1_ltd_seen[widx])
                                        {
                                            int dt = t - int(last_post_conv1[oc][i][j]);
                                            if (dt > 0 && dt < T_STEPS)
                                            {
                                                dw_t dw = stdp_ltd(dt);
                                                rw_conv1_w[widx] = conv_ltd(rw_conv1_w[widx], dw);
                                                conv1_ltd_seen[widx] = ap_uint<1>(1);
                                                diag_inc(diag_stdp_conv1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            commit_post_conv1:
                for (int oc = 0; oc < C1; oc++)
                {
                    for (int i = 0; i < IMG_H; i++)
                    {
                        for (int j = 0; j < IMG_W; j++)
                        {
#pragma HLS PIPELINE II = 1
                            if (spk1[oc][i][j])
                                last_post_conv1[oc][i][j] = ts_t(t);
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
                        }
                    }
                }

            if (enable_conv_stdp)
            {
            stdp_conv2:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
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
                                            if (spk2[oc][i][j])
                                            {
                                                if (p1[ic][ii][jj] && !conv2_ltp_seen[widx])
                                                {
                                                    rw_conv2_w[widx] = conv_ltp(rw_conv2_w[widx], STDP_A_PLUS);
                                                    conv2_ltp_seen[widx] = ap_uint<1>(1);
                                                    diag_inc(diag_stdp_conv2);
                                                }
                                                else if (last_pre_conv2[ic][ii][jj] != TS_NONE && !conv2_ltp_seen[widx])
                                                {
                                                    int dt = t - int(last_pre_conv2[ic][ii][jj]);
                                                    if (dt > 0 && dt < T_STEPS)
                                                    {
                                                        dw_t dw = stdp_ltp(dt);
                                                        rw_conv2_w[widx] = conv_ltp(rw_conv2_w[widx], dw);
                                                        conv2_ltp_seen[widx] = ap_uint<1>(1);
                                                        diag_inc(diag_stdp_conv2);
                                                    }
                                                }
                                            }
                                            if (p1[ic][ii][jj] && last_post_conv2[oc][i][j] != TS_NONE && !conv2_ltd_seen[widx])
                                            {
                                                int dt = t - int(last_post_conv2[oc][i][j]);
                                                if (dt > 0 && dt < T_STEPS)
                                                {
                                                    dw_t dw = stdp_ltd(dt);
                                                    rw_conv2_w[widx] = conv_ltd(rw_conv2_w[widx], dw);
                                                    conv2_ltd_seen[widx] = ap_uint<1>(1);
                                                    diag_inc(diag_stdp_conv2);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            commit_post_conv2:
                for (int oc = 0; oc < C2; oc++)
                {
                    for (int i = 0; i < P1_H; i++)
                    {
                        for (int j = 0; j < P1_W; j++)
                        {
#pragma HLS PIPELINE II = 1
                            if (spk2[oc][i][j])
                                last_post_conv2[oc][i][j] = ts_t(t);
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
                            {
                                last_pre_fc[oc][i][j] = ts_t(t);
                                const int idx = ((oc * P2_H + i) * P2_W + j);
                                fc_seen[idx] = ap_uint<1>(1);
                                if (fc_count[idx] != ap_uint<4>(15))
                                    fc_count[idx]++;
                            }
                        }
                    }
                }
                }
            }

        bp_reset_feedback_c2:
            for (int oc = 0; oc < C2; oc++)
            {
#pragma HLS PIPELINE II = 1
                bp_feedback_c2[oc] = ap_fixed<24, 12>(0);
            }
        bp_reset_feedback_c1:
            for (int ic = 0; ic < C1; ic++)
            {
#pragma HLS PIPELINE II = 1
                bp_feedback_c1[ic] = ap_fixed<24, 12>(0);
            }

            int pred_label = 0;
            ap_fixed<28, 16> best_score = ap_fixed<28, 16>(-32768);

        bp_score_train_sample:
            for (int o = 0; o < FC_OUT; o++)
            {
                ap_fixed<24, 12> active_score = ap_fixed<24, 12>(0);
                ap_fixed<24, 12> row_sum = ap_fixed<24, 12>(0);
                ap_uint<10> active_count = ap_uint<10>(0);
                for (int idx = 0; idx < FC_IN; idx++)
                {
#pragma HLS PIPELINE II = 1
                    ap_fixed<24, 12> w = ap_fixed<24, 12>(rw_fc_w[idx_fc(o, idx)]);
                    row_sum += w;
                    if (fc_seen[idx])
                    {
                        active_score += w;
                        active_count++;
                    }
                }
                ap_fixed<28, 16> mean_active =
                    (ap_fixed<28, 16>(row_sum) * ap_fixed<28, 16>(active_count)) /
                    ap_fixed<28, 16>(FC_IN);
                ap_fixed<28, 16> centered_score =
                    ap_fixed<28, 16>(active_score) - mean_active;
                if (o == 0 || centered_score > best_score)
                {
                    best_score = centered_score;
                    pred_label = o;
                }
            }

            const bool bp_correct = (pred_label == int(train_label));
            if (bp_correct)
                diag_inc(diag_correct_pre);
            else
                diag_inc(diag_wrong_pre);

        bp_active_feature_count:
            for (int idx = 0; idx < FC_IN; idx++)
            {
#pragma HLS PIPELINE II = 1
                if (fc_seen[idx])
                    diag_inc(diag_active_features);
            }

        bp_feedback_from_fc:
            for (int oc = 0; oc < C2; oc++)
            {
                for (int i = 0; i < P2_H; i++)
                {
                    for (int j = 0; j < P2_W; j++)
                    {
#pragma HLS PIPELINE II = 1
                        const int idx = ((oc * P2_H + i) * P2_W + j);
                        if (fc_seen[idx] && !bp_correct)
                        {
                            ap_fixed<24, 12> fb =
                                ap_fixed<24, 12>(rw_fc_w[idx_fc(int(train_label), idx)]);
                            fb -= ap_fixed<24, 12>(rw_fc_w[idx_fc(pred_label, idx)]);
                            bp_feedback_c2[oc] += fb;
                        }
                    }
                }
            }

        bp_feedback_to_conv1:
            for (int oc = 0; oc < C2; oc++)
            {
                for (int ic = 0; ic < C1; ic++)
                {
                    for (int ki = 0; ki < K; ki++)
                    {
                        for (int kj = 0; kj < K; kj++)
                        {
#pragma HLS PIPELINE II = 1
                            int widx = idx_conv2(oc, ic, ki, kj);
                            bp_feedback_c1[ic] +=
                                bp_feedback_c2[oc] * ap_fixed<24, 12>(rw_conv2_w[widx]);
                        }
                    }
                }
            }

        bp_stdp_fc_update:
            for (int o = 0; o < FC_OUT; o++)
            {
                for (int idx = 0; idx < FC_IN; idx++)
                {
#pragma HLS PIPELINE II = 1
                    int widx = idx_fc(o, idx);
                    if (o == int(train_label))
                    {
                        w_t delta = -FC_INACTIVE_LTD;
                        if (fc_seen[idx])
                        {
                            delta = FC_ACTIVE_LTP + (bp_correct ? FC_BP_CORRECT_LTP : FC_BP_ACTIVE_LTP);
                            diag_inc(diag_bp_fc_ltp);
                        }
                        else
                        {
                            diag_inc(diag_bp_fc_ltd);
                        }
                        rw_fc_w[widx] = clamp_w(rw_fc_w[widx] + delta);
                    }
                    else if (fc_seen[idx] && !bp_correct && o == pred_label)
                    {
                        rw_fc_w[widx] = clamp_w(rw_fc_w[widx] - (FC_ANTILABEL_LTD + FC_BP_WRONG_LTD));
                        diag_inc(diag_bp_fc_ltd);
                    }
                    else if (fc_seen[idx])
                    {
                        rw_fc_w[widx] = clamp_w(rw_fc_w[widx] - FC_ANTILABEL_LTD);
                        diag_inc(diag_bp_fc_ltd);
                    }
                }
            }

        bp_stdp_conv2_update:
            for (int oc = 0; oc < C2; oc++)
            {
                for (int ic = 0; ic < C1; ic++)
                {
                    for (int ki = 0; ki < K; ki++)
                    {
                        for (int kj = 0; kj < K; kj++)
                        {
#pragma HLS PIPELINE II = 1
                            int widx = idx_conv2(oc, ic, ki, kj);
                            if (conv2_ltp_seen[widx] || conv2_ltd_seen[widx])
                            {
                                if (bp_feedback_c2[oc] > ap_fixed<24, 12>(0.00390625))
                                {
                                    rw_conv2_w[widx] = conv_ltp(rw_conv2_w[widx], CONV_BP_A_PLUS);
                                    diag_inc(diag_bp_conv2);
                                }
                                else if (bp_feedback_c2[oc] < ap_fixed<24, 12>(-0.00390625))
                                {
                                    rw_conv2_w[widx] = conv_ltd(rw_conv2_w[widx], CONV_BP_A_MINUS);
                                    diag_inc(diag_bp_conv2);
                                }
                            }
                        }
                    }
                }
            }

        bp_stdp_conv1_update:
            for (int oc = 0; oc < C1; oc++)
            {
                for (int ki = 0; ki < K; ki++)
                {
                    for (int kj = 0; kj < K; kj++)
                    {
#pragma HLS PIPELINE II = 1
                        int widx = idx_conv1(oc, ki, kj);
                        if (conv1_ltp_seen[widx] || conv1_ltd_seen[widx])
                        {
                            if (bp_feedback_c1[oc] > ap_fixed<24, 12>(0.00390625))
                            {
                                rw_conv1_w[widx] = conv_ltp(rw_conv1_w[widx], CONV_BP_A_PLUS);
                                diag_inc(diag_bp_conv1);
                            }
                            else if (bp_feedback_c1[oc] < ap_fixed<24, 12>(-0.00390625))
                            {
                                rw_conv1_w[widx] = conv_ltd(rw_conv1_w[widx], CONV_BP_A_MINUS);
                                diag_inc(diag_bp_conv1);
                            }
                        }
                    }
                }
            }
        conv1_homeostasis:
            for (int widx = 0; widx < CONV1_W_SIZE; widx++)
            {
#pragma HLS PIPELINE II = 1
                rw_conv1_w[widx] = conv_homeo(rw_conv1_w[widx], conv_home_target(init_conv1_weight(widx)));
            }
        conv2_homeostasis:
            for (int widx = 0; widx < CONV2_W_SIZE; widx++)
            {
#pragma HLS PIPELINE II = 1
                rw_conv2_w[widx] = conv_homeo(rw_conv2_w[widx], conv_home_target(init_conv2_weight(widx)));
            }
        }
    }

    if (infer_mode)
    {
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
infer_reset_fc_seen:
    for (int i = 0; i < FC_IN; i++)
    {
#pragma HLS PIPELINE II = 1
        fc_seen[i] = ap_uint<1>(0);
        fc_count[i] = ap_uint<4>(0);
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
                        if (v)
                        {
                            const int idx = ((oc * P2_H + i) * P2_W + j);
                            fc_seen[idx] = ap_uint<1>(1);
                            if (fc_count[idx] != ap_uint<4>(15))
                                fc_count[idx]++;
                        }
                    }
                }
            }

            int fc_winner = -1;
            mem_t fc_best = mem_t(-128);

        fc_lif_infer_eval:
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
                fc_next[o] = m;
                if (fc_winner < 0 || m > fc_best)
                {
                    fc_best = m;
                    fc_winner = o;
                }
            }

        fc_lif_infer_commit:
            for (int o = 0; o < FC_OUT; o++)
            {
#pragma HLS PIPELINE II = 1
                ap_uint<1> s = (o == fc_winner) ? ap_uint<1>(1) : ap_uint<1>(0);
                mem3[o] = s ? mem_t(fc_next[o] - fc_v_th) : mem_t(fc_next[o]);
                spk3[o] = s;
                spike_cnt[o] += s;
            }
        }
    }
output_scores:
        for (int o = 0; o < FC_OUT; o++)
        {
            ap_uint<16> q = (ap_uint<16>(spike_cnt[o]) * 256) / T_STEPS;
            if (weighted_mode)
            {
                ap_fixed<24, 12> active_score = ap_fixed<24, 12>(0);
                ap_fixed<24, 12> row_sum = ap_fixed<24, 12>(0);
                ap_fixed<24, 12> active_mass = ap_fixed<24, 12>(0);
                for (int idx = 0; idx < FC_IN; idx++)
                {
                    ap_fixed<24, 12> w = ap_fixed<24, 12>(rw_fc_w[idx_fc(o, idx)]);
                    row_sum += w;
                    if (fc_count[idx] != ap_uint<4>(0))
                    {
                        ap_fixed<24, 12> mass = ap_fixed<24, 12>(1.0);
                        if (fc_count[idx] > ap_uint<4>(1))
                        {
                            mass += ap_fixed<24, 12>(fc_count[idx] - ap_uint<4>(1)) *
                                    ap_fixed<24, 12>(0.25);
                        }
                        active_score += w * mass;
                        active_mass += mass;
                    }
                }
                ap_fixed<28, 16> mean_active =
                    (ap_fixed<28, 16>(row_sum) * ap_fixed<28, 16>(active_mass)) /
                    ap_fixed<28, 16>(FC_IN);
                ap_fixed<28, 16> centered_score =
                    ap_fixed<28, 16>(active_score) - mean_active;
                ap_fixed<28, 16> scaled =
                    ap_fixed<28, 16>(2048.0) +
                    centered_score * ap_fixed<28, 16>(8.0) +
                    ap_fixed<28, 16>(score_bias(o));
                if (scaled <= ap_fixed<28, 16>(0))
                    q = ap_uint<16>(0);
                else if (scaled >= ap_fixed<28, 16>(4095))
                    q = ap_uint<16>(4095);
                else
                    q = ap_uint<16>(scaled);
            }
            const bool last_score = (!weighted_mode && o == FC_OUT - 1);
            write_u16(out_stream, q, last_score);
        }
    }

    if (weighted_mode)
    {
        if (train_mode)
        {
            write_u16(out_stream, ap_uint<16>(0xB57D), false);
            write_u16(out_stream, diag_correct_pre, false);
            write_u16(out_stream, diag_wrong_pre, false);
            write_u16(out_stream, diag_active_features, false);
            write_u16(out_stream, diag_stdp_conv1, false);
            write_u16(out_stream, diag_stdp_conv2, false);
            write_u16(out_stream, diag_bp_conv1, false);
            write_u16(out_stream, diag_bp_conv2, false);
            write_u16(out_stream, diag_bp_fc_ltp, false);
            write_u16(out_stream, diag_bp_fc_ltd, false);
        }
dump_conv1:
        for (int i = 0; i < CONV1_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            write_u16(out_stream, encode_weight_word(rw_conv1_w[i]), false);
        }
dump_conv2:
        for (int i = 0; i < CONV2_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            write_u16(out_stream, encode_weight_word(rw_conv2_w[i]), false);
        }
dump_fc:
        for (int i = 0; i < FC_W_SIZE; i++)
        {
#pragma HLS PIPELINE II = 1
            const bool last_weight = (i == FC_W_SIZE - 1);
            write_u16(out_stream, encode_weight_word(rw_fc_w[i]), last_weight);
        }
    }
}
