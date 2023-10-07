# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        dtype = input.dtype
        # this is a hack, lets use the clip val as the max input for static activation quantization
        if clip_val is not None:
            max_input = max(abs(clip_val[0]), abs(clip_val[1]))
        elif layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input).detach().to(torch.float32)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach().to(torch.float32)
                )
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach().to(torch.float32)
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = (input * s).to(torch.float32)
        output = torch.round(input * s).div(s + 1e-6)

        return output.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        dtype = input.dtype
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static

        # this is a hack, lets use the clip val to indicate the range for static activation quantization
        if clip_val is not None:
            alpha = (clip_val[1] - clip_val[0])
            beta  = clip_val[0]
        elif layerwise:
            alpha = (input.max() - input.min()).detach().to(torch.float32)
            beta  = input.min().detach().to(torch.float32)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach().to(torch.float32)
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach().to(torch.float32)
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach().to(torch.float32)
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach().to(torch.float32)
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        input_normalized = input_normalized.to(torch.float32)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizerWrapper(nn.Module):
    def __init__(self, bitwidth, symmetric, layerwise, clip_val=None):
        super(QuantizerWrapper, self).__init__()

        self.bitwidth  = bitwidth
        self.symmetric = symmetric
        self.layerwise = layerwise
        self.clip_val  = clip_val

        if symmetric:
            self.quantizer = SymQuantizer
        else:
            self.quantizer = AsymQuantizer
    
    def forward(self, input_):
        if self.bitwidth >= 32:
            return input_
        elif self.bitwidth >= 3:
            return self.quantizer.apply(input_, self.clip_val, self.bitwidth, self.layerwise)
        else:
            raise ValueError("bitwidth must be >= 3")


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        bias=False,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
        act_symmetric=False,
        weight_symmetric=False,
        quantize_output=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.input_quantizer  = None
        self.output_quantizer = None
        self.weight_quantizer = None
        self.a_bits = a_bits
        self.w_bits = w_bits
        if self.a_bits < 32 and self.a_bits > 2:
            self.input_quantizer  = QuantizerWrapper(a_bits, act_symmetric, act_layerwise)
            if quantize_output:
                self.output_quantizer = QuantizerWrapper(a_bits, act_symmetric, act_layerwise)
        if self.w_bits < 32 and self.w_bits > 2:
            self.weight_quantizer = QuantizerWrapper(w_bits, weight_symmetric, weight_layerwise)

    def set_act_scale(self, act_scale):
        if self.input_quantizer is not None:
            self.input_quantizer.clip_val = (act_scale['input'][0], act_scale['input'][1])
        if self.output_quantizer is not None:
            self.output_quantizer.clip_val = (act_scale['output'][0], act_scale['output'][1])

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 3:
            weight = self.weight_quantizer(real_weights)
        else:
            raise ValueError("bitwidth must be >= 3")

        # quantize inputs
        if self.a_bits < 32 and self.a_bits > 2:
            input_ = self.input_quantizer(input_)

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out


class QuantizeBMM(nn.Module):
    def __init__(self, 
        a_bits=32, b_bits=32, 
        a_symmetric=False, b_symmetric=False, 
        a_per_tensor_quant=True, b_per_tensor_quant=True, 
        quantize_output=True):
        super().__init__()

        self.a_quantizer = None
        self.b_quantizer = None
        self.output_quantizer = None
        if a_bits < 32 and a_bits > 2:
            self.a_quantizer = QuantizerWrapper(a_bits, a_symmetric, a_per_tensor_quant)
        if b_bits < 32 and b_bits > 2:
            self.b_quantizer = QuantizerWrapper(b_bits, b_symmetric, b_per_tensor_quant)
        if quantize_output:
            self.output_quantizer = QuantizerWrapper(a_bits, a_symmetric, a_per_tensor_quant)

    def set_act_scale(self, act_scale):
        if self.a_quantizer is not None:
            self.a_quantizer.clip_val      = (act_scale['input'][0], act_scale['input'][1])
        if self.b_quantizer is not None:
            self.b_quantizer.clip_val      = (act_scale['input2'][0], act_scale['input2'][1])
        if self.output_quantizer is not None:
            self.output_quantizer.clip_val = (act_scale['output'][0], act_scale['output'][1])
        

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if self.a_quantizer is not None:
            a = self.a_quantizer(a)
        if self.b_quantizer is not None:
            b = self.b_quantizer(b)
        out = torch.matmul(a, b)
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out