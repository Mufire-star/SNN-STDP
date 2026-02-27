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

    unsigned char label0 = 0;
    lblf.read(reinterpret_cast<char *>(&label0), 1);

    std::vector<unsigned char> image(rows * cols);
    imgf.read(reinterpret_cast<char *>(image.data()), image.size());

    hls::stream<axis_in_t> in_stream;
    hls::stream<axis_out_t> out_stream;

    for (int i = 0; i < 28 * 28; i++) {
        axis_in_t p;
        p.data = image[i];
        p.keep = -1;
        p.strb = -1;
        p.last = (i == 28 * 28 - 1) ? 1 : 0;
        in_stream.write(p);
    }

    snn_top(in_stream, out_stream);

    int best_idx = -1;
    int best_val = -1;

    std::cout << "[HW] label=" << int(label0) << " scores=";
    for (int i = 0; i < FC_OUT; i++) {
        axis_out_t o = out_stream.read();
        int v = int(o.data);
        std::cout << v;
        if (i + 1 < FC_OUT) std::cout << ",";
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    std::cout << " pred=" << best_idx << "\n";

    return 0;
}
