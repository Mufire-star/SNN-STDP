import argparse
from pathlib import Path

import torch
import math


def to_c_array_1d(tensor, cast='w_t'):
    vals = tensor.reshape(-1).tolist()
    return '{' + ', '.join(f'{cast}({float(v):.10g})' for v in vals) + '}'


def to_c_array_2d(tensor, cast='w_t'):
    rows = []
    for r in tensor:
        rows.append('{' + ', '.join(f'{cast}({float(v):.10g})' for v in r.tolist()) + '}')
    return '{' + ',\n '.join(rows) + '}'


def to_c_array_4d(tensor, cast='w_t'):
    blocks = []
    for a in tensor:
        c3 = []
        for b in a:
            c2 = []
            for c in b:
                c2.append('{' + ', '.join(f'{cast}({float(v):.10g})' for v in c.tolist()) + '}')
            c3.append('{' + ', '.join(c2) + '}')
        blocks.append('{' + ',\n  '.join(c3) + '}')
    return '{' + ',\n '.join(blocks) + '}'


def to_c_array_flat(tensor, cast='w_t'):
    vals = tensor.reshape(-1).tolist()
    return '{' + ', '.join(f'{cast}({float(v):.10g})' for v in vals) + '}'


def main():
    parser = argparse.ArgumentParser(description='Export trained SNN weights to true_snn_ip headers')
    parser.add_argument('--ckpt', type=str, default=str(Path('python/SNN-baseline/weights/fp32/mnist_snn_baseline.pt')))
    parser.add_argument('--out', type=str, default=str(Path('HW/baseline/snn_ip/weights/weights_generated.h')))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['state_dict']

    conv1_w = sd['net.0.weight']
    bn1_g = sd['net.1.weight']
    bn1_b = sd['net.1.bias']
    bn1_m = sd['net.1.running_mean']
    bn1_v = sd['net.1.running_var']

    conv2_w = sd['net.4.weight']
    bn2_g = sd['net.5.weight']
    bn2_b = sd['net.5.bias']
    bn2_m = sd['net.5.running_mean']
    bn2_v = sd['net.5.running_var']

    fc_w = sd['net.8.weight'] if 'net.8.weight' in sd else sd['net.9.weight']
    fc_b = sd['net.8.bias'] if 'net.8.bias' in sd else sd['net.9.bias']

    # Fold BN into conv weights/biases: y = gamma*(x-mean)/sqrt(var+eps)+beta
    eps = 1e-5
    bn1_scale = bn1_g / torch.sqrt(bn1_v + eps)
    conv1_w_fold = conv1_w * bn1_scale.view(-1, 1, 1, 1)
    conv1_b_fold = bn1_b - bn1_scale * bn1_m

    bn2_scale = bn2_g / torch.sqrt(bn2_v + eps)
    conv2_w_fold = conv2_w * bn2_scale.view(-1, 1, 1, 1)
    conv2_b_fold = bn2_b - bn2_scale * bn2_m

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = f'''#ifndef TRUE_SNN_WEIGHTS_GENERATED_H_
#define TRUE_SNN_WEIGHTS_GENERATED_H_

#include "../snn_top.h"

static const w_t conv1_w[C1 * 1 * K * K] = {to_c_array_flat(conv1_w_fold)};
static const w_t conv1_b[C1] = {to_c_array_1d(conv1_b_fold)};

static const w_t conv2_w[C2 * C1 * K * K] = {to_c_array_flat(conv2_w_fold)};
static const w_t conv2_b[C2] = {to_c_array_1d(conv2_b_fold)};

static const w_t fc_w[FC_OUT * FC_IN] = {to_c_array_flat(fc_w)};
static const w_t fc_b[FC_OUT] = {to_c_array_1d(fc_b)};

#endif
'''
    out_path.write_text(content)
    print(f'Exported true SNN weights to: {out_path}')


if __name__ == '__main__':
    main()
