#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "snn_top.h"

static uint32_t read_be_u32(std::ifstream &ifs)
{
    unsigned char b[4];
    ifs.read(reinterpret_cast<char *>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

static std::vector<std::string> mnist_candidates(const char *basename)
{
    std::vector<std::string> out;
    if (const char *raw_dir = std::getenv("MNIST_RAW_DIR"))
    {
        out.emplace_back(std::string(raw_dir) + "/" + basename);
    }

    const char *prefixes[] = {
        "data/mnist/raw/",
        "../data/mnist/raw/",
        "../../data/mnist/raw/",
        "../../../data/mnist/raw/",
        "../../../../data/mnist/raw/",
        "../../../../../data/mnist/raw/",
        "../../../../../../data/mnist/raw/",
    };

    for (const char *prefix : prefixes)
    {
        out.emplace_back(std::string(prefix) + basename);
    }
    return out;
}

static bool open_first_existing(std::ifstream &ifs, std::string &resolved_path, const std::vector<std::string> &candidates)
{
    for (const auto &candidate : candidates)
    {
        ifs.open(candidate, std::ios::binary);
        if (ifs.is_open())
        {
            resolved_path = candidate;
            return true;
        }
        ifs.clear();
    }
    return false;
}

static int collect_scores(
    hls::stream<axis_out_t> &out_stream,
    std::array<int, FC_OUT> &scores,
    const char *tag,
    bool final_score_has_tlast = true,
    bool require_empty = true)
{
    int best_idx = -1;
    int best_val = -1;
    bool all_zero = true;

    for (int i = 0; i < FC_OUT; i++)
    {
        if (out_stream.empty())
        {
            std::cerr << tag << ": output stream ended early at class " << i << "\n";
            return -1;
        }

        axis_out_t o = out_stream.read();
        scores[i] = int(o.data);
        all_zero = all_zero && (scores[i] == 0);

        const bool expected_last = final_score_has_tlast && (i == FC_OUT - 1);
        if (expected_last != bool(o.last))
        {
            std::cerr << tag << ": TLAST mismatch at class " << i << "\n";
            return -1;
        }

        if (scores[i] > best_val)
        {
            best_val = scores[i];
            best_idx = i;
        }
    }

    if (require_empty && !out_stream.empty())
    {
        std::cerr << tag << ": output stream has unexpected extra words\n";
        return -1;
    }

    if (all_zero)
    {
        std::cerr << tag << ": accelerator returned all-zero scores\n";
    }

    return best_idx;
}

static void push_byte(hls::stream<axis_in_t> &in_stream, unsigned char value, bool last = false)
{
    axis_in_t p;
    p.data = value;
    p.keep = -1;
    p.strb = -1;
    p.last = last ? 1 : 0;
    in_stream.write(p);
}

static void push_s8(hls::stream<axis_in_t> &in_stream, int8_t value)
{
    push_byte(in_stream, static_cast<unsigned char>(value));
}

static int8_t encode_weight(float value)
{
    float scaled = value * 128.0f;
    int raw = scaled >= 0.0f ? int(scaled + 0.5f) : int(scaled - 0.5f);
    if (raw > 127)
        raw = 127;
    if (raw < -128)
        raw = -128;
    return static_cast<int8_t>(raw);
}

static std::vector<int8_t> initial_weight_words()
{
    std::vector<int8_t> weights(TOTAL_WEIGHT_WORDS);
    int cursor = 0;

    for (int idx = 0; idx < CONV1_W_SIZE; idx++)
    {
        const int bucket = (idx * 5 + 7) % 11;
        const float values[] = {0.035f, 0.039f, 0.043f, 0.047f, 0.051f, 0.055f, 0.059f, 0.063f, 0.067f, 0.071f, 0.075f};
        weights[cursor++] = encode_weight(values[bucket]);
    }
    for (int idx = 0; idx < CONV2_W_SIZE; idx++)
    {
        const int bucket = (idx * 13 + 5) % 6;
        const float values[] = {0.020f, 0.027f, 0.034f, 0.041f, 0.048f, 0.055f};
        weights[cursor++] = encode_weight(values[bucket]);
    }
    for (int idx = 0; idx < FC_W_SIZE; idx++)
    {
        weights[cursor++] = encode_weight(0.0f);
    }

    return weights;
}

static void push_weights(hls::stream<axis_in_t> &in_stream, const std::vector<int8_t> &weights)
{
    for (int i = 0; i < TOTAL_WEIGHT_WORDS; i++)
    {
        push_s8(in_stream, weights[i]);
    }
}

static int collect_weights(
    hls::stream<axis_out_t> &out_stream,
    std::vector<int8_t> &weights,
    const char *tag)
{
    weights.assign(TOTAL_WEIGHT_WORDS, 0);
    for (int i = 0; i < TOTAL_WEIGHT_WORDS; i++)
    {
        if (out_stream.empty())
        {
            std::cerr << tag << ": output stream ended early at weight " << i << "\n";
            return -1;
        }
        axis_out_t o = out_stream.read();
        weights[i] = static_cast<int8_t>(static_cast<unsigned char>(o.data & 0x00FF));
        const bool expected_last = (i == TOTAL_WEIGHT_WORDS - 1);
        if (expected_last != bool(o.last))
        {
            std::cerr << tag << ": TLAST mismatch at weight " << i << "\n";
            return -1;
        }
    }
    if (!out_stream.empty())
    {
        std::cerr << tag << ": output stream has unexpected extra words\n";
        return -1;
    }
    return 0;
}

static int run_infer(
    hls::stream<axis_in_t> &in_stream,
    hls::stream<axis_out_t> &out_stream,
    const std::vector<unsigned char> &image,
    std::array<int, FC_OUT> &scores)
{
    axis_in_t mode_p;
    mode_p.data = MODE_INFER;
    mode_p.keep = -1;
    mode_p.strb = -1;
    mode_p.last = 0;
    in_stream.write(mode_p);

    for (int i = 0; i < IMG_H * IMG_W; i++)
    {
        axis_in_t p;
        p.data = image[i];
        p.keep = -1;
        p.strb = -1;
        p.last = 0;
        in_stream.write(p);
    }

    snn_top(in_stream, out_stream);
    return collect_scores(out_stream, scores, "infer");
}

static int run_train_then_infer(
    hls::stream<axis_in_t> &in_stream,
    hls::stream<axis_out_t> &out_stream,
    const std::vector<std::vector<unsigned char>> &train_images,
    const std::vector<unsigned char> &train_labels,
    const std::vector<unsigned char> &test_image,
    std::array<int, FC_OUT> &scores)
{
    axis_in_t mode_p;
    mode_p.data = MODE_TRAIN;
    mode_p.keep = -1;
    mode_p.strb = -1;
    mode_p.last = 0;
    in_stream.write(mode_p);

    for (int n = 0; n < NUM_TRAIN_IMG; n++)
    {
        const auto &img = train_images[n];
        axis_in_t label_p;
        label_p.data = train_labels[n];
        label_p.keep = -1;
        label_p.strb = -1;
        label_p.last = 0;
        in_stream.write(label_p);

        for (int i = 0; i < IMG_H * IMG_W; i++)
        {
            axis_in_t p;
            p.data = img[i];
            p.keep = -1;
            p.strb = -1;
            p.last = 0;
            in_stream.write(p);
        }
    }

    for (int i = 0; i < IMG_H * IMG_W; i++)
    {
        axis_in_t p;
        p.data = test_image[i];
        p.keep = -1;
        p.strb = -1;
        p.last = (i == IMG_H * IMG_W - 1) ? 1 : 0;
        in_stream.write(p);
    }

    snn_top(in_stream, out_stream);
    return collect_scores(out_stream, scores, "train_then_infer");
}

static int run_weighted_train_only(
    hls::stream<axis_in_t> &in_stream,
    hls::stream<axis_out_t> &out_stream,
    const std::vector<int8_t> &weights_in,
    const std::vector<std::vector<unsigned char>> &train_images,
    const std::vector<unsigned char> &train_labels,
    std::vector<int8_t> &weights_out)
{
    push_byte(in_stream, MODE_WEIGHTED_TRAIN_ONLY);
    push_weights(in_stream, weights_in);

    for (int n = 0; n < NUM_TRAIN_IMG; n++)
    {
        push_byte(in_stream, train_labels[n]);
        for (int i = 0; i < IMG_H * IMG_W; i++)
        {
            push_byte(in_stream, train_images[n][i]);
        }
    }

    snn_top(in_stream, out_stream);
    return collect_weights(out_stream, weights_out, "weighted_train_only");
}

static int run_weighted_infer(
    hls::stream<axis_in_t> &in_stream,
    hls::stream<axis_out_t> &out_stream,
    const std::vector<int8_t> &weights,
    const std::vector<unsigned char> &image,
    std::array<int, FC_OUT> &scores,
    std::vector<int8_t> &returned_weights)
{
    push_byte(in_stream, MODE_WEIGHTED_INFER);
    push_weights(in_stream, weights);
    for (int i = 0; i < IMG_H * IMG_W; i++)
    {
        push_byte(in_stream, image[i], i == IMG_H * IMG_W - 1);
    }

    snn_top(in_stream, out_stream);
    const int pred = collect_scores(out_stream, scores, "weighted_infer", false, false);
    if (pred < 0)
        return -1;
    if (collect_weights(out_stream, returned_weights, "weighted_infer") != 0)
        return -1;
    return pred;
}

int main()
{
    std::ifstream imgf;
    std::ifstream lblf;
    std::string img_path;
    std::string lbl_path;

    if (!open_first_existing(imgf, img_path, mnist_candidates("t10k-images-idx3-ubyte")) ||
        !open_first_existing(lblf, lbl_path, mnist_candidates("t10k-labels-idx1-ubyte")))
    {
        std::cerr << "Failed to open MNIST raw files. Set MNIST_RAW_DIR or download data to data/mnist/raw.\n";
        return 1;
    }

    std::cout << "Using images: " << img_path << "\n";
    std::cout << "Using labels: " << lbl_path << "\n";

    const uint32_t magic_i = read_be_u32(imgf);
    const uint32_t num_i = read_be_u32(imgf);
    const uint32_t rows = read_be_u32(imgf);
    const uint32_t cols = read_be_u32(imgf);

    const uint32_t magic_l = read_be_u32(lblf);
    const uint32_t num_l = read_be_u32(lblf);

    if (magic_i != 2051 || magic_l != 2049 || rows != IMG_H || cols != IMG_W || num_i == 0 || num_l == 0)
    {
        std::cerr << "Invalid MNIST headers.\n";
        return 1;
    }

    const int total_need = NUM_TRAIN_IMG + 1;
    if (num_i < static_cast<uint32_t>(total_need) || num_l < static_cast<uint32_t>(total_need))
    {
        std::cerr << "Not enough MNIST samples for testbench.\n";
        return 1;
    }

    std::vector<unsigned char> labels(total_need);
    std::vector<std::vector<unsigned char>> images(total_need, std::vector<unsigned char>(rows * cols));

    for (int n = 0; n < total_need; n++)
    {
        lblf.read(reinterpret_cast<char *>(&labels[n]), 1);
        imgf.read(reinterpret_cast<char *>(images[n].data()), rows * cols);
    }

    hls::stream<axis_in_t> in_stream;
    hls::stream<axis_out_t> out_stream;
    std::array<int, FC_OUT> baseline_scores = {};
    std::array<int, FC_OUT> trained_scores = {};

    const int pred_baseline = run_infer(in_stream, out_stream, images[0], baseline_scores);

    std::vector<std::vector<unsigned char>> train_imgs(images.begin() + 1, images.begin() + 1 + NUM_TRAIN_IMG);
    std::vector<unsigned char> train_lbls(labels.begin() + 1, labels.begin() + 1 + NUM_TRAIN_IMG);
    const int pred_after_train = run_train_then_infer(in_stream, out_stream, train_imgs, train_lbls, images[0], trained_scores);

    if (pred_baseline < 0 || pred_after_train < 0)
    {
        std::cerr << "Testbench detected invalid accelerator output.\n";
        return 2;
    }

    std::cout << "\n=== STDP C Simulation Summary ===\n";
    std::cout << "Test label:             " << int(labels[0]) << "\n";
    std::cout << "Inference only result:  " << pred_baseline << "\n";
    std::cout << "Inference scores:       ";
    for (int i = 0; i < FC_OUT; i++)
    {
        std::cout << baseline_scores[i] << (i + 1 == FC_OUT ? '\n' : ' ');
    }
    std::cout << "Train-then-infer result:" << pred_after_train << "\n";
    std::cout << "Train-then-infer scores:";
    for (int i = 0; i < FC_OUT; i++)
    {
        std::cout << " " << trained_scores[i];
    }
    std::cout << "\n";
    std::cout << "Training labels:";
    for (int i = 0; i < NUM_TRAIN_IMG; i++)
    {
        std::cout << " " << int(labels[1 + i]);
    }
    std::cout << "\n";

    if (pred_after_train == int(labels[0]))
    {
        std::cout << "NOTE: STDP-updated run matched the test label.\n";
    }
    else
    {
        std::cout << "NOTE: This is an unsupervised STDP demo, so exact classification is not guaranteed.\n";
    }

    std::vector<int8_t> weights_before = initial_weight_words();
    std::array<int, FC_OUT> echo_scores = {};
    std::vector<int8_t> echo_weights;
    const int pred_echo = run_weighted_infer(in_stream, out_stream, weights_before, images[0], echo_scores, echo_weights);
    if (pred_echo < 0)
    {
        std::cerr << "Weighted infer echo path failed.\n";
        return 3;
    }

    int echo_mismatches = 0;
    int first_echo_mismatch = -1;
    for (int i = 0; i < TOTAL_WEIGHT_WORDS; i++)
    {
        if (weights_before[i] != echo_weights[i])
        {
            if (first_echo_mismatch < 0)
                first_echo_mismatch = i;
            echo_mismatches++;
        }
    }
    if (echo_mismatches != 0)
    {
        std::cerr << "Weighted infer did not preserve PS-supplied weights; mismatches="
                  << echo_mismatches << ", first=" << first_echo_mismatch
                  << ", sent=" << int(weights_before[first_echo_mismatch])
                  << ", returned=" << int(echo_weights[first_echo_mismatch]) << "\n";
        return 4;
    }

    std::vector<int8_t> weights_after;
    if (run_weighted_train_only(in_stream, out_stream, weights_before, train_imgs, train_lbls, weights_after) != 0)
    {
        std::cerr << "Weighted train-only path failed.\n";
        return 5;
    }

    int changed_total = 0;
    int changed_conv1 = 0;
    int changed_conv2 = 0;
    int changed_fc = 0;
    const int fc_start = CONV1_W_SIZE + CONV2_W_SIZE;
    const int conv2_start = CONV1_W_SIZE;
    for (int i = 0; i < TOTAL_WEIGHT_WORDS; i++)
    {
        if (weights_before[i] != weights_after[i])
        {
            changed_total++;
            if (i < conv2_start)
                changed_conv1++;
            else if (i < fc_start)
                changed_conv2++;
            else
                changed_fc++;
        }
    }

    std::array<int, FC_OUT> weighted_scores = {};
    std::vector<int8_t> returned_weights;
    const int pred_weighted = run_weighted_infer(in_stream, out_stream, weights_after, images[0], weighted_scores, returned_weights);
    if (pred_weighted < 0)
    {
        std::cerr << "Weighted infer path failed.\n";
        return 6;
    }

    std::cout << "\n=== Weighted Persistence Protocol Summary ===\n";
    std::cout << "Changed weights total:  " << changed_total << "\n";
    std::cout << "Changed Conv1 weights:  " << changed_conv1 << "\n";
    std::cout << "Changed Conv2 weights:  " << changed_conv2 << "\n";
    std::cout << "Changed FC weights:     " << changed_fc << "\n";
    std::cout << "Weighted infer result:  " << pred_weighted << "\n";
    std::cout << "Weighted infer scores:  ";
    for (int i = 0; i < FC_OUT; i++)
    {
        std::cout << weighted_scores[i] << (i + 1 == FC_OUT ? '\n' : ' ');
    }

    if (changed_fc == 0)
    {
        std::cerr << "Weighted train-only did not change any FC weights.\n";
        return 7;
    }

    return 0;
}
