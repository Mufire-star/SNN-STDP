#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "snn_top.h"

static uint32_t read_be_u32(std::ifstream &ifs) {
    unsigned char b[4];
    ifs.read(reinterpret_cast<char *>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

static int run_infer(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream,
                     const std::vector<unsigned char> &image) {
    axis_in_t mode_p;
    mode_p.data = MODE_INFER;
    mode_p.keep = -1;
    mode_p.strb = -1;
    mode_p.last = 0;
    in_stream.write(mode_p);

    for (int i = 0; i < 28 * 28; i++) {
        axis_in_t p;
        p.data = image[i];
        p.keep = -1;
        p.strb = -1;
        p.last = 0;
        in_stream.write(p);
    }

    snn_top(in_stream, out_stream);

    int best_idx = -1;
    int best_val = -1;
    int scores[FC_OUT] = {0};

    for (int i = 0; i < FC_OUT; i++) {
        axis_out_t o = out_stream.read();
        scores[i] = int(o.data);
        if (scores[i] > best_val) {
            best_val = scores[i];
            best_idx = i;
        }
    }

    return best_idx;
}

static int run_train_then_infer(hls::stream<axis_in_t> &in_stream, hls::stream<axis_out_t> &out_stream,
                                const std::vector<std::vector<unsigned char>> &train_images,
                                const std::vector<unsigned char> &test_image) {
    axis_in_t mode_p;
    mode_p.data = MODE_TRAIN;
    mode_p.keep = -1;
    mode_p.strb = -1;
    mode_p.last = 0;
    in_stream.write(mode_p);

    for (int n = 0; n < NUM_TRAIN_IMG; n++) {
        const auto &img = train_images[n];
        for (int i = 0; i < 28 * 28; i++) {
            axis_in_t p;
            p.data = img[i];
            p.keep = -1;
            p.strb = -1;
            p.last = 0;
            in_stream.write(p);
        }
    }

    for (int i = 0; i < 28 * 28; i++) {
        axis_in_t p;
        p.data = test_image[i];
        p.keep = -1;
        p.strb = -1;
        p.last = (i == 28 * 28 - 1) ? 1 : 0;
        in_stream.write(p);
    }

    snn_top(in_stream, out_stream);

    int best_idx = -1;
    int best_val = -1;
    int scores[FC_OUT] = {0};

    for (int i = 0; i < FC_OUT; i++) {
        axis_out_t o = out_stream.read();
        scores[i] = int(o.data);
        if (scores[i] > best_val) {
            best_val = scores[i];
            best_idx = i;
        }
    }

    return best_idx;
}

int main() {
    const std::vector<std::string> img_candidates = {
        "/home/distortionk/WorkSpace/VCS/SNN-STDP/python/SNN-baseline/datas/MNIST/raw/t10k-images-idx3-ubyte",
        "../../python/SNN-baseline/datas/MNIST/raw/t10k-images-idx3-ubyte",
        "../../../python/SNN-baseline/datas/MNIST/raw/t10k-images-idx3-ubyte",
    };
    const std::vector<std::string> lbl_candidates = {
        "/home/distortionk/WorkSpace/VCS/SNN-STDP/python/SNN-baseline/datas/MNIST/raw/t10k-labels-idx1-ubyte",
        "../../python/SNN-baseline/datas/MNIST/raw/t10k-labels-idx1-ubyte",
        "../../../python/SNN-baseline/datas/MNIST/raw/t10k-labels-idx1-ubyte",
    };

    std::ifstream imgf;
    std::ifstream lblf;
    for (const auto &p : img_candidates) {
        imgf.open(p, std::ios::binary);
        if (imgf.is_open()) break;
    }
    for (const auto &p : lbl_candidates) {
        lblf.open(p, std::ios::binary);
        if (lblf.is_open()) break;
    }
    if (!imgf.is_open() || !lblf.is_open()) {
        std::cerr << "Failed to open MNIST files.\n";
        return 1;
    }

    uint32_t magic_i = read_be_u32(imgf);
    uint32_t num_i = read_be_u32(imgf);
    uint32_t rows = read_be_u32(imgf);
    uint32_t cols = read_be_u32(imgf);

    uint32_t magic_l = read_be_u32(lblf);
    uint32_t num_l = read_be_u32(lblf);

    if (magic_i != 2051 || magic_l != 2049 || rows != 28 || cols != 28 || num_i == 0 || num_l == 0) {
        std::cerr << "Invalid MNIST headers.\n";
        return 1;
    }

    const int total_need = NUM_TRAIN_IMG + 2;
    if (num_i < total_need || num_l < total_need) {
        std::cerr << "Not enough MNIST images.\n";
        return 1;
    }

    std::vector<unsigned char> labels(total_need);
    std::vector<std::vector<unsigned char>> images(total_need, std::vector<unsigned char>(rows * cols));

    for (int n = 0; n < total_need; n++) {
        lblf.read(reinterpret_cast<char *>(&labels[n]), 1);
        imgf.read(reinterpret_cast<char *>(images[n].data()), rows * cols);
    }

    hls::stream<axis_in_t> in_stream;
    hls::stream<axis_out_t> out_stream;

    std::cout << "=== Baseline Inference (no training) ===\n";
    int pred_baseline = run_infer(in_stream, out_stream, images[0]);
    std::cout << "[INFER] label=" << int(labels[0]) << " pred=" << pred_baseline << "\n\n";

    std::cout << "=== Train+Infer (STDP on " << NUM_TRAIN_IMG << " images) ===\n";
    std::vector<std::vector<unsigned char>> train_imgs(images.begin() + 1, images.begin() + 1 + NUM_TRAIN_IMG);
    std::cout << "Training images labels:";
    for (int i = 0; i < NUM_TRAIN_IMG; i++) {
        std::cout << " " << int(labels[1 + i]);
    }
    std::cout << "\n";

    int pred_after_train = run_train_then_infer(in_stream, out_stream, train_imgs, images[0]);
    std::cout << "[TRAIN+INFER] test_label=" << int(labels[0]) << " pred=" << pred_after_train << "\n\n";

    std::cout << "=== Summary ===\n";
    std::cout << "Test image label:     " << int(labels[0]) << "\n";
    std::cout << "Baseline prediction:  " << pred_baseline << "\n";
    std::cout << "After STDP prediction:" << pred_after_train << "\n";

    if (pred_after_train == int(labels[0])) {
        std::cout << "PASS: STDP training result matches label.\n";
    } else if (pred_baseline == int(labels[0])) {
        std::cout << "NOTE: Baseline was correct but STDP changed it (unsupervised STDP may not preserve accuracy).\n";
    } else {
        std::cout << "NOTE: Neither baseline nor STDP matched label (expected for unsupervised STDP on few images).\n";
    }

    return 0;
}
