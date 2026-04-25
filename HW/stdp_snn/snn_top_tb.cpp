#include <cstdlib>
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
    const char *tag)
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

        if ((i == FC_OUT - 1) != bool(o.last))
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

    if (!out_stream.empty())
    {
        std::cerr << tag << ": output stream has unexpected extra words\n";
        return -1;
    }

    if (all_zero)
    {
        std::cerr << tag << ": accelerator returned all-zero scores\n";
        return -1;
    }

    return best_idx;
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

    const int total_need = NUM_TRAIN_IMG + 2;
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
    const int pred_after_train = run_train_then_infer(in_stream, out_stream, train_imgs, images[0], trained_scores);

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

    return 0;
}
