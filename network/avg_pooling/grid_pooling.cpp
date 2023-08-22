#include <torch/extension.h>
#include "utils.h"


torch::Tensor avg_pooling_forward(
    torch::Tensor point_feat,
    torch::Tensor pidx,
    torch::Tensor pidx_counts,
    int grid_size)
{
    CHECK_INPUT(point_feat);
    CHECK_INPUT(pidx);
    CHECK_INPUT(pidx_counts);
    return avg_pooling_forward_cu(point_feat, pidx, pidx_counts, grid_size);
};

torch::Tensor avg_pooling_backward(
    torch::Tensor grad_output,
    torch::Tensor pidx,
    torch::Tensor pidx_counts)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pidx);
    CHECK_INPUT(pidx_counts);
    return avg_pooling_backward_cu(grad_output, pidx, pidx_counts);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("avg_pooling_forward", &avg_pooling_forward);
    m.def("avg_pooling_backward", &avg_pooling_backward);
};