#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <tuple>
#include <iostream>
#include "gbpn_cuda.h"

void GBPLR_F(
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int j) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPLR_F_Cuda(beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       j);
}

void GBPLR_B(
        at::Tensor &doutput,
        at::Tensor &dconf,
        at::Tensor &doutput9,
        at::Tensor &dconf9,
        at::Tensor &dbeta,
        at::Tensor &dpost,
        at::Tensor &dmask,
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int j) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPLR_B_Cuda(doutput,
                       dconf,
                       doutput9,
                       dconf9,
                       dbeta,
                       dpost,
                       dmask,
                       input,
                       prior,
                       beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       j);
}

void GBPTD_F(
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int i) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPTD_F_Cuda(beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       i);
}

void GBPTD_B(
        at::Tensor &doutput,
        at::Tensor &dconf,
        at::Tensor &doutput9,
        at::Tensor &dconf9,
        at::Tensor &dbeta,
        at::Tensor &dpost,
        at::Tensor &dmask,
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int i) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPTD_B_Cuda(doutput,
                       dconf,
                       doutput9,
                       dconf9,
                       dbeta,
                       dpost,
                       dmask,
                       input,
                       prior,
                       beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       i);
}

void GBPRL_F(
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int j) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPRL_F_Cuda(beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       j);
}

void GBPRL_B(
        at::Tensor &doutput,
        at::Tensor &dconf,
        at::Tensor &doutput9,
        at::Tensor &dconf9,
        at::Tensor &dbeta,
        at::Tensor &dpost,
        at::Tensor &dmask,
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int j) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPRL_B_Cuda(doutput,
                       dconf,
                       doutput9,
                       dconf9,
                       dbeta,
                       dpost,
                       dmask,
                       input,
                       prior,
                       beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       j);
}

void GBPDT_F(
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int i) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPDT_F_Cuda(beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       i);
}


void GBPDT_B(
        at::Tensor &doutput,
        at::Tensor &dconf,
        at::Tensor &doutput9,
        at::Tensor &dconf9,
        at::Tensor &dbeta,
        at::Tensor &dpost,
        at::Tensor &dmask,
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &mask,
        at::Tensor &output,
        at::Tensor &conf,
        at::Tensor &output9,
        at::Tensor &conf9,
        int i) {
    int batch_size = output.size(0);
    int height = output.size(2);
    int width = output.size(3);
    GBPDT_B_Cuda(doutput,
                       dconf,
                       doutput9,
                       dconf9,
                       dbeta,
                       dpost,
                       dmask,
                       input,
                       prior,
                       beta,
                       post,
                       mask,
                       output,
                       conf,
                       output9,
                       conf9,
                       batch_size,
                       height,
                       width,
                       i);
}

void GBPNL_F(
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &offs,
        at::Tensor &output,
        at::Tensor &conf) {
    int batch_size = offs.size(0);
    int off_size = offs.size(1) / 2;
    int height = offs.size(2);
    int width = offs.size(3);
    GBPNL_F_Cuda(input,
                     prior,
                     beta,
                     post,
                     offs,
                     output,
                     conf,
                     batch_size,
                     off_size,
                     height,
                     width);
}


void GBPNL_B(
        at::Tensor &doutput,
        at::Tensor &dconf,
        at::Tensor &dinput,
        at::Tensor &dprior,
        at::Tensor &dbeta,
        at::Tensor &dpost,
        at::Tensor &doffs,
        at::Tensor &input,
        at::Tensor &prior,
        at::Tensor &beta,
        at::Tensor &post,
        at::Tensor &offs,
        at::Tensor &output,
        at::Tensor &conf) {
    int batch_size = offs.size(0);
    int off_size = offs.size(1) / 2;
    int height = offs.size(2);
    int width = offs.size(3);
    GBPNL_B_Cuda(doutput,
                     dconf,
                     dinput,
                     dprior,
                     dbeta,
                     dpost,
                     doffs,
                     input,
                     prior,
                     beta,
                     post,
                     offs,
                     output,
                     conf,
                     batch_size,
                     off_size,
                     height,
                     width);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("GBPLR_F", &GBPLR_F, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPLR_B", &GBPLR_B, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPTD_F", &GBPTD_F, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPTD_B", &GBPTD_B, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPRL_F", &GBPRL_F, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPRL_B", &GBPRL_B, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPDT_F", &GBPDT_F, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPDT_B", &GBPDT_B, "Gaussian Belief Propagation Layer New Damping 3 direction");
m.def("GBPNL_F", &GBPNL_F, "Gaussian Belief Propagation Layer New Damping");
m.def("GBPNL_B", &GBPNL_B, "Gaussian Belief Propagation Layer New Damping");
}
