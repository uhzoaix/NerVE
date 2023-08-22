#include <torch/extension.h>
#include <stdio.h>
#include <iostream>

__constant__ float grid_size = 1.0;

template <typename scalar_t>
__global__ void avg_pooling_kernel_fw(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> point_feat,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out_feat,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> pidx,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> pidx_counts,
    const int n)
{
    int c = threadIdx.x;
    for (int p=0;p < n;p++){
        int idx = pidx[p];
        out_feat[idx][c] += point_feat[p][c] / float(pidx_counts[idx]);
    }
}

/*
average pooling on points with their feature

params:
    point_feat: (N, C)
    points: (N, 3)
    k: grid size
output:
    out_feat: (k,k,k, C)
*/
torch::Tensor avg_pooling_forward_cu(
    torch::Tensor point_feat,
    torch::Tensor pidx,
    torch::Tensor pidx_counts,
    int k)
{
    const int N = point_feat.size(0), C = point_feat.size(1);

    torch::Tensor out_feat = torch::zeros({k * k * k, C}, point_feat.options());
    // options: torch::dtype(torch::kInt32).device(points.device)

    int blocks = 1;
    int threds = C;

    AT_DISPATCH_FLOATING_TYPES(point_feat.type(), "avg_pooling_forward_cu",
                               ([&]
                                { avg_pooling_kernel_fw<scalar_t><<<blocks, threds>>>(
                                      point_feat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      out_feat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                      pidx_counts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                      N); }));

    return out_feat;
}

template <typename scalar_t>
__global__ void avg_pooling_kernel_bw(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out_grad,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> pidx,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> pidx_counts,
    const int n)
{
    int c = threadIdx.x;
    for (int p = 0; p < n; p++)
    {
        int idx = pidx[p];
        out_grad[p][c] = grad_output[idx][c] / float(pidx_counts[idx]);
    }
}

/*
average pooling on points with their feature

params:
    grad_output: (k**3, C)
    points: (N, 3)
    k: grid size
output:
    out_feat: (N, C)
*/
torch::Tensor avg_pooling_backward_cu(
    torch::Tensor grad_output,
    torch::Tensor pidx,
    torch::Tensor pidx_counts)
{
    const int N = pidx.size(0);
    const int C = grad_output.size(1);

    torch::Tensor out_grad = torch::zeros({N, C}, grad_output.options());
    // options: torch::dtype(torch::kInt32).device(points.device)

    int blocks = 1;
    int threds = C;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "avg_pooling_backward_cu",
                               ([&]
                                { avg_pooling_kernel_bw<scalar_t><<<blocks, threds>>>(
                                      grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      out_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                      pidx_counts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                      N); }));

    return out_grad;
}