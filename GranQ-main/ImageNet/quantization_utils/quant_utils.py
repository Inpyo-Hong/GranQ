#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import numpy as np
from torch.autograd import Function, Variable
import torch




def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_point: shift for quantization
    """

    B = input.shape[0]
    C = input.shape[1]

    if len(input.shape) == 4:
        scale = scale.view(B, C, 1, 1)  # [B, C, 1, 1]
        zero_point = zero_point.view(B, C, 1, 1)
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_point: shift for quantization
    """
    B = input.shape[0]
    C = input.shape[1]

    if len(input.shape) == 4:
        scale = scale.view(B, C, 1, 1)  # [B, C, 1, 1]
        zero_point = zero_point.view(B, C, 1, 1)

    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """


    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        if x.dim() == 4:  # conv layer [batch, channel, width, height]

            min_val = x.view(x.size(0), x.size(1), -1).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            max_val = x.view(x.size(0), x.size(1), -1).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)

            scale, zero_point = asymmetric_linear_quantization_params(k, min_val, max_val)

            new_quant_x = linear_quantize(x, scale, zero_point)

            quant_x = linear_dequantize(new_quant_x, scale, zero_point)

            quantized_x = quant_x

        else:  # Fully connected layers
            min_val = x.min(dim=1, keepdim=True)[0]
            max_val = x.max(dim=1, keepdim=True)[0]

            scale, zero_point = asymmetric_linear_quantization_params(k, min_val, max_val)

            new_quant_x = linear_quantize(x, scale, zero_point)
            quant_x = linear_dequantize(new_quant_x, scale, zero_point)

            quantized_x = quant_x


        return torch.autograd.Variable(quantized_x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
