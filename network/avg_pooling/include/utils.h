#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


torch::Tensor avg_pooling_forward_cu(
    torch::Tensor point_feat,
    torch::Tensor pidx,
    torch::Tensor pidx_counts,
    int k);

torch::Tensor avg_pooling_backward_cu(
    torch::Tensor grad_output,
    torch::Tensor pidx,
    torch::Tensor pidx_counts
);