#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include "gbpn_cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>

    template<typename scalar_t>
    __global__ void
    gbplr_f_kernel(scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tn) {

        const int32_t OFF[3][2] = {{-1, -1},
                                   {0,  -1},
                                   {1,  -1}};
        const int32_t MAP[3] = {0, 3, 5};

        int32_t thm = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tm = num * blockDim.x + thm;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s;

        if ((tm >= 0) && (tm < height)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        conf_s = mask_s * conf_pre + (1 - mask_s) * conf_s;
                        oc_s = mask_s * out_pre + (1 - mask_s) * oc_s;
                    }
                    output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = oc_s;
                    atomicAdd(output9 + batch_index * height * width + tm * width + tn, oc_s - out_pre);
                    conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = conf_s;
                    atomicAdd(conf9 + batch_index * height * width + tm * width + tn, conf_s - conf_pre);
                }
            }
        }
    }

    template<typename scalar_t>
    __global__ void
    gbplr_b_kernel(scalar_t *__restrict__ doutput, scalar_t *__restrict__ dconf,
                         scalar_t *__restrict__ doutput9, scalar_t *__restrict__ dconf9,
                         scalar_t *__restrict__ dbeta, scalar_t *__restrict__ dpost, scalar_t *__restrict__ dmask,
                         scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                         scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tn) {

        const int32_t OFF[3][2] = {{-1, -1},
                                   {0,  -1},
                                   {1,  -1}};
        const int32_t MAP[3] = {0, 3, 5};

        int32_t thm = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tm = num * blockDim.x + thm;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s,
                doutput_t, doutput_s, dconf_t, dconf_s;

        if ((tm >= 0) && (tm < height)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = prior[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = input[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    doutput_t = doutput9[batch_index * height * width + tm * width + tn] +
                                doutput[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    dconf_t = dconf9[batch_index * height * width + tm * width + tn] +
                              dconf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        atomicAdd(dmask + batch_index * height * width + tm * width + tn,
                                  dconf_t * (conf_pre - conf_s) + doutput_t * (out_pre - oc_s));
                        atomicAdd(
                                doutput + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                doutput_t * (mask_s - 1));
                        doutput_t = doutput_t * (1 - mask_s);
                        atomicAdd(
                                dconf + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                dconf_t * (mask_s - 1));
                        dconf_t = dconf_t * (1 - mask_s);
                    }
                    doutput_s = doutput_t * (post_s / (prior_s + post_s));
                    dbeta[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            doutput_t * conf_s;
                    dconf_t = dconf_t + doutput_t * (ip_s / prior_s + db_s);
                    dpost[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            dconf_t * (prior_s / (prior_s + post_s)) * (prior_s / (prior_s + post_s));
                    dconf_s = dconf_t * (post_s / (prior_s + post_s)) * (post_s / (prior_s + post_s)) -
                              doutput_t * (conf_s / prior_s) * (ip_s / prior_s);
                    atomicAdd(doutput9 + batch_index * height * width + sm * width + sn, doutput_s);
                    atomicAdd(doutput + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -doutput_s);
                    atomicAdd(dconf9 + batch_index * height * width + sm * width + sn, dconf_s);
                    atomicAdd(dconf + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -dconf_s);
                }
            }
        }
    }

    template<typename scalar_t>
    __global__ void
    gbptd_f_kernel(scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tm) {

        const int32_t OFF[3][2] = {{-1, -1},
                                   {-1, 0},
                                   {-1, 1}};
        const int32_t MAP[3] = {0, 1, 2};

        int32_t thn = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tn = num * blockDim.x + thn;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s;

        if ((tn >= 0) && (tn < width)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        conf_s = mask_s * conf_pre + (1 - mask_s) * conf_s;
                        oc_s = mask_s * out_pre + (1 - mask_s) * oc_s;
                    }
                    output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = oc_s;
                    atomicAdd(output9 + batch_index * height * width + tm * width + tn, oc_s - out_pre);
                    conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = conf_s;
                    atomicAdd(conf9 + batch_index * height * width + tm * width + tn, conf_s - conf_pre);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbptd_b_kernel(scalar_t *__restrict__ doutput, scalar_t *__restrict__ dconf,
                         scalar_t *__restrict__ doutput9, scalar_t *__restrict__ dconf9,
                         scalar_t *__restrict__ dbeta, scalar_t *__restrict__ dpost, scalar_t *__restrict__ dmask,
                         scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                         scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tm) {

        const int32_t OFF[3][2] = {{-1, -1},
                                   {-1, 0},
                                   {-1, 1}};
        const int32_t MAP[3] = {0, 1, 2};

        int32_t thn = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tn = num * blockDim.x + thn;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s,
                doutput_t, doutput_s, dconf_t, dconf_s;

        if ((tn >= 0) && (tn < width)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = prior[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = input[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    doutput_t = doutput9[batch_index * height * width + tm * width + tn] +
                                doutput[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    dconf_t = dconf9[batch_index * height * width + tm * width + tn] +
                              dconf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        atomicAdd(dmask + batch_index * height * width + tm * width + tn,
                                  dconf_t * (conf_pre - conf_s) + doutput_t * (out_pre - oc_s));
                        atomicAdd(
                                doutput + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                doutput_t * (mask_s - 1));
                        doutput_t = doutput_t * (1 - mask_s);
                        atomicAdd(
                                dconf + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                dconf_t * (mask_s - 1));
                        dconf_t = dconf_t * (1 - mask_s);
                    }
                    doutput_s = doutput_t * (post_s / (prior_s + post_s));
                    dbeta[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            doutput_t * conf_s;
                    dconf_t = dconf_t + doutput_t * (ip_s / prior_s + db_s);
                    dpost[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            dconf_t * (prior_s / (prior_s + post_s)) * (prior_s / (prior_s + post_s));
                    dconf_s = dconf_t * (post_s / (prior_s + post_s)) * (post_s / (prior_s + post_s)) -
                              doutput_t * (conf_s / prior_s) * (ip_s / prior_s);
                    atomicAdd(doutput9 + batch_index * height * width + sm * width + sn, doutput_s);
                    atomicAdd(doutput + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -doutput_s);
                    atomicAdd(dconf9 + batch_index * height * width + sm * width + sn, dconf_s);
                    atomicAdd(dconf + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -dconf_s);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbprl_f_kernel(scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tn) {

        const int32_t OFF[3][2] = {{-1, 1},
                                   {0,  1},
                                   {1,  1}};
        const int32_t MAP[3] = {2, 4, 7};

        int32_t thm = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tm = num * blockDim.x + thm;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s;

        if ((tm >= 0) && (tm < height)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        conf_s = mask_s * conf_pre + (1 - mask_s) * conf_s;
                        oc_s = mask_s * out_pre + (1 - mask_s) * oc_s;
                    }
                    output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = oc_s;
                    atomicAdd(output9 + batch_index * height * width + tm * width + tn, oc_s - out_pre);
                    conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = conf_s;
                    atomicAdd(conf9 + batch_index * height * width + tm * width + tn, conf_s - conf_pre);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbprl_b_kernel(scalar_t *__restrict__ doutput, scalar_t *__restrict__ dconf,
                         scalar_t *__restrict__ doutput9, scalar_t *__restrict__ dconf9,
                         scalar_t *__restrict__ dbeta, scalar_t *__restrict__ dpost, scalar_t *__restrict__ dmask,
                         scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                         scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tn) {

        const int32_t OFF[3][2] = {{-1, 1},
                                   {0,  1},
                                   {1,  1}};
        const int32_t MAP[3] = {2, 4, 7};

        int32_t thm = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tm = num * blockDim.x + thm;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s,
                doutput_t, doutput_s, dconf_t, dconf_s;

        if ((tm >= 0) && (tm < height)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = prior[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = input[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    doutput_t = doutput9[batch_index * height * width + tm * width + tn] +
                                doutput[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    dconf_t = dconf9[batch_index * height * width + tm * width + tn] +
                              dconf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        atomicAdd(dmask + batch_index * height * width + tm * width + tn,
                                  dconf_t * (conf_pre - conf_s) + doutput_t * (out_pre - oc_s));
                        atomicAdd(
                                doutput + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                doutput_t * (mask_s - 1));
                        doutput_t = doutput_t * (1 - mask_s);
                        atomicAdd(
                                dconf + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                dconf_t * (mask_s - 1));
                        dconf_t = dconf_t * (1 - mask_s);
                    }
                    doutput_s = doutput_t * (post_s / (prior_s + post_s));
                    dbeta[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            doutput_t * conf_s;
                    dconf_t = dconf_t + doutput_t * (ip_s / prior_s + db_s);
                    dpost[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            dconf_t * (prior_s / (prior_s + post_s)) * (prior_s / (prior_s + post_s));
                    dconf_s = dconf_t * (post_s / (prior_s + post_s)) * (post_s / (prior_s + post_s)) -
                              doutput_t * (conf_s / prior_s) * (ip_s / prior_s);
                    atomicAdd(doutput9 + batch_index * height * width + sm * width + sn, doutput_s);
                    atomicAdd(doutput + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -doutput_s);
                    atomicAdd(dconf9 + batch_index * height * width + sm * width + sn, dconf_s);
                    atomicAdd(dconf + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -dconf_s);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbpdt_f_kernel(scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tm) {

        const int32_t OFF[8][2] = {{1, -1},
                                   {1, 0},
                                   {1, 1}};
        const int32_t MAP[3] = {5, 6, 7};

        int32_t thn = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tn = num * blockDim.x + thn;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s;

        if ((tn >= 0) && (tn < width)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        conf_s = mask_s * conf_pre + (1 - mask_s) * conf_s;
                        oc_s = mask_s * out_pre + (1 - mask_s) * oc_s;
                    }
                    output[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = oc_s;
                    atomicAdd(output9 + batch_index * height * width + tm * width + tn, oc_s - out_pre);
                    conf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn] = conf_s;
                    atomicAdd(conf9 + batch_index * height * width + tm * width + tn, conf_s - conf_pre);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbpdt_b_kernel(scalar_t *__restrict__ doutput, scalar_t *__restrict__ dconf,
                         scalar_t *__restrict__ doutput9, scalar_t *__restrict__ dconf9,
                         scalar_t *__restrict__ dbeta, scalar_t *__restrict__ dpost, scalar_t *__restrict__ dmask,
                         scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                         scalar_t *__restrict__ beta, scalar_t *__restrict__ post, scalar_t *__restrict__ mask,
                         scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                         scalar_t *__restrict__ output9, scalar_t *__restrict__ conf9,
                         size_t batch_size, size_t height, size_t width, size_t tm) {


        const int32_t OFF[8][2] = {{1, -1},
                                   {1, 0},
                                   {1, 1}};
        const int32_t MAP[3] = {5, 6, 7};

        int32_t thn = threadIdx.x;
        int32_t num1, num2;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        int32_t num = num1 / 3;
        int32_t pp = num1 - num * 3;

        int32_t tn = num * blockDim.x + thn;
        int32_t sm = tm + OFF[pp][0];
        int32_t sn = tn + OFF[pp][1];
        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s,
                doutput_t, doutput_s, dconf_t, dconf_s;

        if ((tn >= 0) && (tn < width)) {
            if ((sm >= 0) && (sm < height) && (sn >= 0) && (sn < width)) {
                ip_s = output9[batch_index * height * width + sm * width + sn] -
                       output[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                prior_s = conf9[batch_index * height * width + sm * width + sn] -
                          conf[batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width + sn];
                if (prior_s > 0) {
                    db_s = beta[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * 3 * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = prior[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    out_pre = input[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    doutput_t = doutput9[batch_index * height * width + tm * width + tn] +
                                doutput[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    dconf_t = dconf9[batch_index * height * width + tm * width + tn] +
                              dconf[batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn];
                    if (conf_pre > 0) {
                        mask_s = mask[batch_index * height * width + tm * width + tn];
                        atomicAdd(dmask + batch_index * height * width + tm * width + tn,
                                  dconf_t * (conf_pre - conf_s) + doutput_t * (out_pre - oc_s));
                        atomicAdd(
                                doutput + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                doutput_t * (mask_s - 1));
                        doutput_t = doutput_t * (1 - mask_s);
                        atomicAdd(
                                dconf + batch_index * 8 * height * width + MAP[pp] * height * width + tm * width + tn,
                                dconf_t * (mask_s - 1));
                        dconf_t = dconf_t * (1 - mask_s);
                    }
                    doutput_s = doutput_t * (post_s / (prior_s + post_s));
                    dbeta[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            doutput_t * conf_s;
                    dconf_t = dconf_t + doutput_t * (ip_s / prior_s + db_s);
                    dpost[batch_index * 3 * height * width + pp * height * width + tm * width + tn] =
                            dconf_t * (prior_s / (prior_s + post_s)) * (prior_s / (prior_s + post_s));
                    dconf_s = dconf_t * (post_s / (prior_s + post_s)) * (post_s / (prior_s + post_s)) -
                              doutput_t * (conf_s / prior_s) * (ip_s / prior_s);
                    atomicAdd(doutput9 + batch_index * height * width + sm * width + sn, doutput_s);
                    atomicAdd(doutput + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -doutput_s);
                    atomicAdd(dconf9 + batch_index * height * width + sm * width + sn, dconf_s);
                    atomicAdd(dconf + batch_index * 8 * height * width + (7 - MAP[pp]) * height * width + sm * width +
                              sn, -dconf_s);
                }
            }
        }
    }

template<typename scalar_t>
    __global__ void
    gbpnl_f_kernel(scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                       scalar_t *__restrict__ beta, scalar_t *__restrict__ post,
                       scalar_t *__restrict__ offs,
                       scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                       size_t batch_size, size_t off_size, size_t height, size_t width) {


        int32_t num1, num2, num3;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        num2 = num1 / off_size;
        int32_t pp = num1 - num2 * off_size;
        num3 = num2 * blockDim.x + threadIdx.x;

        int32_t tm = num3 / width;
        int32_t tn = num3 - tm * width;

        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre, mask_s;
        scalar_t ip_s1 = 0, ip_s2 = 0, ip_s3 = 0, ip_s4 = 0;
        scalar_t prior_s1 = 0, prior_s2 = 0, prior_s3 = 0, prior_s4 = 0;


        if ((tm >= 0) && (tm < height)) {
            scalar_t sm = tm +
                          offs[batch_index * off_size * 2 * height * width + pp * 2 * height * width +
                               0 * height * width +
                               tm * width + tn];
            scalar_t sn = tn +
                          offs[batch_index * off_size * 2 * height * width + pp * 2 * height * width +
                               1 * height * width +
                               tm * width + tn];
            int32_t m_low = floor(sm);
            int32_t n_low = floor(sn);
            int32_t m_high = m_low + 1;
            int32_t n_high = n_low + 1;
            scalar_t lm = sm - m_low;
            scalar_t ln = sn - n_low;
            scalar_t hm = 1 - lm;
            scalar_t hn = 1 - ln;
            scalar_t w1 = hm * hn, w2 = hm * ln, w3 = lm * hn, w4 = lm * ln;

            if ((sm > -1) && (sm < height) && (sn > -1) && (sn < width)) {
                if (m_low >= 0 && n_low >= 0) {
                    ip_s1 = input[batch_index * height * width + m_low * width + n_low];
                    prior_s1 = prior[batch_index * height * width + m_low * width + n_low];
                }
                if (m_low >= 0 && n_high < width) {
                    ip_s2 = input[batch_index * height * width + m_low * width + n_high];
                    prior_s2 = prior[batch_index * height * width + m_low * width + n_high];
                }
                if (m_high < height && n_low >= 0) {
                    ip_s3 = input[batch_index * height * width + m_high * width + n_low];
                    prior_s3 = prior[batch_index * height * width + m_high * width + n_low];
                }
                if (m_high < height && n_high < width) {
                    ip_s4 = input[batch_index * height * width + m_high * width + n_high];
                    prior_s4 = prior[batch_index * height * width + m_high * width + n_high];
                }
                ip_s = w1 * ip_s1 + w2 * ip_s2 + w3 * ip_s3 + w4 * ip_s4;
                prior_s = w1 * prior_s1 + w2 * prior_s2 + w3 * prior_s3 + w4 * prior_s4;
                if (prior_s > 0) {
                    db_s = beta[batch_index * off_size * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * off_size * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    atomicAdd(output + batch_index * height * width + tm * width + tn, oc_s);
                    atomicAdd(conf + batch_index * height * width + tm * width + tn, conf_s);
                }
            }
        }
    }


    template<typename scalar_t>
    __global__ void
    gbpnl_b_kernel(scalar_t *__restrict__ doutput, scalar_t *__restrict__ dconf,
                       scalar_t *__restrict__ dinput, scalar_t *__restrict__ dprior,
                       scalar_t *__restrict__ dbeta, scalar_t *__restrict__ dpost,
                       scalar_t *__restrict__ doffs,
                       scalar_t *__restrict__ input, scalar_t *__restrict__ prior,
                       scalar_t *__restrict__ beta, scalar_t *__restrict__ post,
                       scalar_t *__restrict__ offs,
                       scalar_t *__restrict__ output, scalar_t *__restrict__ conf,
                       size_t batch_size, size_t off_size, size_t height, size_t width) {

        int32_t num1, num2, num3;
        num1 = blockIdx.x / batch_size;
        int32_t batch_index = blockIdx.x - num1 * batch_size;;
        num2 = num1 / off_size;
        int32_t pp = num1 - num2 * off_size;
        num3 = num2 * blockDim.x + threadIdx.x;

        int32_t tm = num3 / width;
        int32_t tn = num3 - tm * width;


        scalar_t ip_s1 = 0, ip_s2 = 0, ip_s3 = 0, ip_s4 = 0;
        scalar_t prior_s1 = 0, prior_s2 = 0, prior_s3 = 0, prior_s4 = 0;

        scalar_t dw1, dw2, dw3, dw4;

        scalar_t depth_s, ip_s, oc_s, db_s, prior_s, post_s, conf_s, out_pre, conf_pre,
                doutput_t, doutput_s, dconf_t, dconf_s;

        if ((tm >= 0) && (tm < height)) {
            scalar_t sm = tm +
                          offs[batch_index * off_size * 2 * height * width + pp * 2 * height * width +
                               0 * height * width +
                               tm * width + tn];
            scalar_t sn = tn +
                          offs[batch_index * off_size * 2 * height * width + pp * 2 * height * width +
                               1 * height * width +
                               tm * width + tn];

            int32_t m_low = floor(sm);
            int32_t n_low = floor(sn);
            int32_t m_high = m_low + 1;
            int32_t n_high = n_low + 1;
            scalar_t lm = sm - m_low;
            scalar_t ln = sn - n_low;
            scalar_t hm = 1 - lm;
            scalar_t hn = 1 - ln;
            scalar_t w1 = hm * hn, w2 = hm * ln, w3 = lm * hn, w4 = lm * ln;

            if ((sm > -1) && (sm < height) && (sn > -1) && (sn < width)) {
                if (m_low >= 0 && n_low >= 0) {
                    ip_s1 = input[batch_index * height * width + m_low * width + n_low];
                    prior_s1 = prior[batch_index * height * width + m_low * width + n_low];
                }
                if (m_low >= 0 && n_high < width) {
                    ip_s2 = input[batch_index * height * width + m_low * width + n_high];
                    prior_s2 = prior[batch_index * height * width + m_low * width + n_high];
                }
                if (m_high < height && n_low >= 0) {
                    ip_s3 = input[batch_index * height * width + m_high * width + n_low];
                    prior_s3 = prior[batch_index * height * width + m_high * width + n_low];
                }
                if (m_high < height && n_high < width) {
                    ip_s4 = input[batch_index * height * width + m_high * width + n_high];
                    prior_s4 = prior[batch_index * height * width + m_high * width + n_high];
                }
                ip_s = w1 * ip_s1 + w2 * ip_s2 + w3 * ip_s3 + w4 * ip_s4;
                prior_s = w1 * prior_s1 + w2 * prior_s2 + w3 * prior_s3 + w4 * prior_s4;
                if (prior_s > 0) {
                    db_s = beta[batch_index * off_size * height * width + pp * height * width + tm * width + tn];
                    post_s = post[batch_index * off_size * height * width + pp * height * width + tm * width + tn];
                    conf_s = post_s * (prior_s / (prior_s + post_s));
                    oc_s = (ip_s / prior_s + db_s) * conf_s;
                    conf_pre = prior[batch_index * height * width + tm * width + tn];
                    doutput_t = doutput[batch_index * height * width + tm * width + tn];
                    dconf_t = dconf[batch_index * height * width + tm * width + tn];
                    doutput_s = doutput_t * (post_s / (prior_s + post_s));
                    dbeta[batch_index * off_size * height * width + pp * height * width + tm * width + tn] =
                            doutput_t * conf_s;
                    dconf_t = dconf_t + doutput_t * (ip_s / prior_s + db_s);
                    dpost[batch_index * off_size * height * width + pp * height * width + tm * width + tn] =
                            dconf_t * (prior_s / (prior_s + post_s)) * (prior_s / (prior_s + post_s));
                    dconf_s = dconf_t * (post_s / (prior_s + post_s)) * (post_s / (prior_s + post_s)) -
                              doutput_t * (conf_s / prior_s) * (ip_s / prior_s);
                    if (m_low >= 0 && n_low >= 0) {
                        atomicAdd(dinput + batch_index * height * width + m_low * width + n_low, doutput_s * w1);
                        atomicAdd(dprior + batch_index * height * width + m_low * width + n_low, dconf_s * w1);
                    }
                    if (m_low >= 0 && n_high < width) {
                        atomicAdd(dinput + batch_index * height * width + m_low * width + n_high, doutput_s * w2);
                        atomicAdd(dprior + batch_index * height * width + m_low * width + n_high, dconf_s * w2);
                    }
                    if (m_high < height && n_low >= 0) {
                        atomicAdd(dinput + batch_index * height * width + m_high * width + n_low, doutput_s * w3);
                        atomicAdd(dprior + batch_index * height * width + m_high * width + n_low, dconf_s * w3);
                    }
                    if (m_high < height && n_high < width) {
                        atomicAdd(dinput + batch_index * height * width + m_high * width + n_high, doutput_s * w4);
                        atomicAdd(dprior + batch_index * height * width + m_high * width + n_high, dconf_s * w4);
                    }

                    dw1 = doutput_s * ip_s1 + dconf_s * prior_s1;
                    dw2 = doutput_s * ip_s2 + dconf_s * prior_s2;
                    dw3 = doutput_s * ip_s3 + dconf_s * prior_s3;
                    dw4 = doutput_s * ip_s4 + dconf_s * prior_s4;
                    doffs[batch_index * off_size * 2 * height * width + pp * 2 * height * width + 0 * height * width +
                          tm * width + tn] = -dw1 * hn - dw2 * ln + dw3 * hn + dw4 * ln;
                    doffs[batch_index * off_size * 2 * height * width + pp * 2 * height * width + 1 * height * width +
                          tm * width + tn] = -dw1 * hm + dw2 * hm - dw3 * lm + dw4 * lm;

                }
            }
        }
    }


void GBPLR_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((height + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbplr_f_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(), output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, j);

}

void GBPLR_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((height + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbplr_b_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            doutput.data_ptr<float>(), dconf.data_ptr<float>(),
                    doutput9.data_ptr<float>(), dconf9.data_ptr<float>(),
                    dbeta.data_ptr<float>(), dpost.data_ptr<float>(), dmask.data_ptr<float>(),
                    input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, j);
}

void GBPTD_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbptd_f_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(), output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, i);

}

void GBPTD_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbptd_b_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            doutput.data_ptr<float>(), dconf.data_ptr<float>(),
                    doutput9.data_ptr<float>(), dconf9.data_ptr<float>(),
                    dbeta.data_ptr<float>(), dpost.data_ptr<float>(), dmask.data_ptr<float>(),
                    input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, i);
}

void GBPRL_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((height + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbprl_f_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(), output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, j);

}

void GBPRL_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t j) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((height + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbprl_b_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            doutput.data_ptr<float>(), dconf.data_ptr<float>(),
                    doutput9.data_ptr<float>(), dconf9.data_ptr<float>(),
                    dbeta.data_ptr<float>(), dpost.data_ptr<float>(), dmask.data_ptr<float>(),
                    input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, j);
}

void GBPDT_F_Cuda(at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbpdt_f_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(), output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, i);

}

void GBPDT_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                        at::Tensor &doutput9, at::Tensor &dconf9,
                        at::Tensor &dbeta, at::Tensor &dpost, at::Tensor &dmask,
                        at::Tensor &input, at::Tensor &prior,
                        at::Tensor &beta, at::Tensor &post, at::Tensor &mask,
                        at::Tensor &output, at::Tensor &conf, at::Tensor &output9, at::Tensor &conf9,
                        size_t batch_size, size_t height, size_t width, size_t i) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * 3 * int((width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbpdt_b_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            doutput.data_ptr<float>(), dconf.data_ptr<float>(),
                    doutput9.data_ptr<float>(), dconf9.data_ptr<float>(),
                    dbeta.data_ptr<float>(), dpost.data_ptr<float>(), dmask.data_ptr<float>(),
                    input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(), mask.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    output9.data_ptr<float>(), conf9.data_ptr<float>(),
                    batch_size, height, width, i);
}

void GBPNL_F_Cuda(at::Tensor &input, at::Tensor &prior,
                      at::Tensor &beta, at::Tensor &post,
                      at::Tensor &offs,
                      at::Tensor &output, at::Tensor &conf,
                      size_t batch_size, size_t off_size, size_t height, size_t width) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * off_size * int((height * width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbpnl_f_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(),
                    offs.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    batch_size, off_size, height, width);

}


void GBPNL_B_Cuda(at::Tensor &doutput, at::Tensor &dconf,
                      at::Tensor &dinput, at::Tensor &dprior,
                      at::Tensor &dbeta, at::Tensor &dpost,
                      at::Tensor &doffs,
                      at::Tensor &input, at::Tensor &prior,
                      at::Tensor &beta, at::Tensor &post,
                      at::Tensor &offs,
                      at::Tensor &output, at::Tensor &conf,
                      size_t batch_size, size_t off_size, size_t height, size_t width) {
    dim3 blockSize(32);
    dim3 gridSize(batch_size * off_size * int((height * width + 32 - 1) / 32));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gbpnl_b_kernel<float>
            << < gridSize, blockSize, 0, stream>>>(
            doutput.data_ptr<float>(), dconf.data_ptr<float>(),
                    dinput.data_ptr<float>(), dprior.data_ptr<float>(),
                    dbeta.data_ptr<float>(), dpost.data_ptr<float>(),
                    doffs.data_ptr<float>(),
                    input.data_ptr<float>(), prior.data_ptr<float>(),
                    beta.data_ptr<float>(), post.data_ptr<float>(),
                    offs.data_ptr<float>(),
                    output.data_ptr<float>(), conf.data_ptr<float>(),
                    batch_size, off_size, height, width);
}