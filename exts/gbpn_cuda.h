#ifndef GBPN_CUDA_H
#define GBPN_CUDA_H
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <tuple>
#include <iostream>

void GBPLR_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j);

void GBPLR_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j);

void GBPTD_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i);

void GBPTD_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i);

void GBPRL_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j);

void GBPRL_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j);

void GBPDT_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i);

void GBPDT_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i);

void GBPNL_F_Cuda(at::Tensor &input, at::Tensor &prior,
                      at::Tensor &beta, at::Tensor &post, at::Tensor &offs,
                      at::Tensor &output, at::Tensor &conf,
                      size_t batch_size, size_t off_size, size_t height, size_t width);


void GBPNL_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                      at::Tensor &dinput, at::Tensor &dprior,
                      at::Tensor &dbeta, at::Tensor &dpost,
                      at::Tensor &doffs,
                      at::Tensor &input, at::Tensor &prior,
                      at::Tensor &beta, at::Tensor &post,
                      at::Tensor &offs,
                      at::Tensor &output, at::Tensor &conf,
                      size_t batch_size, size_t off_size, size_t height, size_t width);

#endif