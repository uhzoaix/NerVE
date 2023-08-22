import torch
from torch.autograd import Function
import grid_pooling


class AvgPoolingModule(torch.nn.Module):
    """
    params:

    """
    def __init__(self, grid_size) -> None:
        super().__init__()
        k = grid_size
        # self.ks = torch.IntTensor([1, k, k**2])
        self.ks = torch.IntTensor([k**2, k, 1])
        self.func = AvgPoolingCuda.apply


    def forward(self, point_feat, points):
        return self.func(point_feat, points, self.ks)


class AvgPoolingCuda(Function):
    """
    params:
        points: (N, 3) in [-1,1]^3
        point_feat: (N, C)
        ks: [1, k, k**2]; k is grid size
    output:
        pooling_feature: (k,k,k, C)
    """
    @staticmethod
    def forward(ctx, point_feat, points, ks):
        C = point_feat.shape[1]
        k = ks[1].item()

        points = (points + 1) *(k/2)
        pidx = torch.floor(points).int() @ ks
        pidx_counts = torch.bincount(pidx).int()
        if point_feat.is_cuda:
            pidx = pidx.to(point_feat.device)
            pidx_counts = pidx_counts.to(point_feat.device)

        out_feat = grid_pooling.avg_pooling_forward(
            point_feat, 
            pidx,
            pidx_counts,
            k
        )
        ctx.pidx_counts = pidx_counts
        ctx.pidx = pidx
        return out_feat.view((k,k,k, C))


    @staticmethod
    def backward(ctx, grad_output):
        pidx_counts, pidx = ctx.pidx_counts, ctx.pidx
        grad_pointfeat = None

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise NotImplementedError('Only calc grad for point_feat')
        
        if len(grad_output.shape) == 4:
            k = grad_output.shape[0]
            C = grad_output.shape[-1]
            grad_output = grad_output.view((k**3, C)).contiguous()
        else:
            raise NotImplementedError

        grad_pointfeat = grid_pooling.avg_pooling_backward(
            grad_output,
            pidx,
            pidx_counts
        )
        
        return grad_pointfeat, None, None


class VerySlowAvgPooling(Function):
    """
    params:
        points: (N, 3) in [-1,1]^3
        point_feat: (N, C)
        ks: [1, k, k**2]; k is grid size
    output:
        pooling_feature: (k,k,k, C)
    """
    @staticmethod
    def forward(ctx, point_feat, points, ks):
        C = point_feat.shape[1]
        k = ks[1].item()
        out_feat = torch.zeros((k**3, C), dtype=point_feat.dtype)

        if point_feat.is_cuda:
            out_feat = out_feat.to(point_feat.device)

        if points.is_cuda:
            points = points.cpu()

        points = (points + 1) *(k/2)
        pidx = points.int() @ ks
        count = {idx.item(): 0 for idx in torch.unique(pidx)}

        for i, idx in enumerate(pidx):
            # if neccessary, record its index for backward
            idx = idx.item()
            out_feat[idx] += point_feat[i]
            count[idx] += 1

        for idx,num in count.items():
            out_feat[idx] /= num

        ctx.count = count
        ctx.pidx = pidx
        return out_feat.view((k,k,k, C))

    @staticmethod
    def backward(ctx, grad_output):
        count, pidx = ctx.count, ctx.pidx
        grad_pointfeat = None

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise NotImplementedError('Only calc grad for point_feat')
        
        if len(grad_output.shape) == 4:
            k = grad_output.shape[0]
            C = grad_output.shape[-1]
            grad_output = grad_output.view((k**3, C))
        else:
            raise NotImplementedError

        N = pidx.shape[0]
        grad_pointfeat = torch.zeros((N, C), device=grad_output.device)
        for i,idx in enumerate(pidx):
            idx = idx.item()
            num = count[idx]
            grad_pointfeat[i] = (1/num)* grad_output[idx]

        return grad_pointfeat, None, None


if __name__ == '__main__':
    import numpy as np
    from time import time
    np.random.seed(2023)
    torch.manual_seed(2023)

    num_pts = 1000
    k = 32
    dim_feat = 128

    points = np.random.uniform(-0.8, 0.8, size=(num_pts, 3))
    points = torch.from_numpy(points)

    point_feat = torch.rand(num_pts, dim_feat).cuda()
    point_feat.requires_grad = True

    avg_pool = AvgPoolingModule(k)
    t0 = time()
    out_feat = avg_pool(point_feat, points)

    loss = torch.sum(out_feat**2) / 2
    loss.backward()
    print(out_feat.shape)
    print(point_feat.grad.shape)
    print('time cost: ', time()-t0)

    