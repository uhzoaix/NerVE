import torch
import torch.nn as nn

from mlp import *
from grid import *


def define_grid_encoder(opt):
    if opt.backbone == 'pointgrid':
        return PointGridEncoder(opt.grid)
    else:
        raise NotImplementedError('Not supported encoder backbone')


class CubeFaceNet(nn.Module):
    """docstring for CubeFaceNet"""
    def __init__(self, opt):
        super().__init__()
        # MLP: [2*Nf, L,L, 1]
        self.mlp = MLP(**opt.model.cube_face)

    def forward(self, cube_feat, feat_3n):
        # cube_feat: (N_pc, N_f)->(N_pc,3,N_f)
        N_pc, N_f = cube_feat.shape
        cube_feat = cube_feat.reshape(N_pc, 1, N_f).repeat(1,3,1)
        feat = torch.cat([cube_feat, feat_3n], dim=-1)

        return self.mlp.forward_simple(feat)


class EdgeGeomNet(nn.Module):
    """docstring for EdgeGeomNet"""
    def __init__(self, opt):
        super(EdgeGeomNet, self).__init__()
        # self.k = opt.model.grid_size
        self.encoder = define_grid_encoder(opt.model.encoder)

        self.decoder_point = MLP(**opt.model.decoder_point)
        
    def forward(self, model_input):
        # batch size is 1
        # cid: (N_cubepc, 3); eid: (N_edgepc, 3)
        k = model_input['info']['grid_size']
        eid = model_input['info']['edge_grid_idx']
        # feat of shape: (k,k,k, Nf)
        geom_feat = self.encoder(model_input)
        # geom_featgrid: (N_edgepc, N_f)
        edge_feat = geom_feat[eid[:,0], eid[:,1],eid[:,2]]
        # out_point: (N_edgepc, 3)
        out_point = self.decoder_point.forward_simple(edge_feat)
        out_point = torch.clamp(out_point, min=-1., max=1.)
        res = {'pc_point': out_point}
        return res


    def forward_simple(self, model_input):
        raise NotImplementedError

    def val_pts_pred(self, out_pts, gt_pts):
        err = torch.sum(torch.abs(out_pts - gt_pts), dim=1).mean()
        return err.item()

    def forward_val(self, model_input, gt):
        k = model_input['info']['grid_size']
        eid = model_input['info']['edge_grid_idx']

        geom_feat = self.encoder(model_input)

        edge_feat = geom_feat[eid[:,0], eid[:,1],eid[:,2]]
        out_point = self.decoder_point.forward_simple(edge_feat)
        torch.clamp_(out_point, min=-1., max=1.)
        pts_err = self.val_pts_pred(out_point, gt['pc_point'])
        pts_dist = torch.norm((out_point-gt['pc_point'])/k, dim=1).mean().item()
        res = {
            'pts_err': pts_err,
            'pts_dist': pts_dist
        }
        return res


    def predict_curve(self, model_input, peid):
        k = model_input['info']['grid_size']
        geom_feat = self.encoder(model_input)

        N_e = peid.shape[0]
        if N_e == 0:
            return None

        edge_feat = geom_feat[peid[:,0], peid[:,1],peid[:,2]]
        out_point = self.decoder_point.forward_simple(edge_feat)
        torch.clamp_(out_point, min=-1., max=1.)

        if out_point.is_cuda:
            peid = peid.to(out_point.device)

        step = 2./ k
        centers = step*peid + (step/2. - 1.)
        out_point = centers + out_point / k

        edge_geom = {'cube_points': out_point.detach().cpu().numpy()}
        return edge_geom



class EdgeCubeNet(nn.Module):
    """docstring for EdgeCubeNet"""
    def __init__(self, opt):
        super(EdgeCubeNet, self).__init__()
        self.encoder = define_grid_encoder(opt.model.encoder)
        self.decoder_cube = MLP(**opt.model.decoder_cube)
        self.predict_type = opt.model.predict_type
        if self.predict_type == 'BCE':
            self.sigmoid = nn.Sigmoid()
        elif self.predict_type == 'NLL':
            self.sigmoid = nn.LogSoftmax(dim=1)
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, model_input):
        # batch size is 1
        # cid: (N_cubepc, 3)
        # k = model_input['info']['grid_size']
        cid = model_input['info']['cube_grid_idx']
        # feat of shape: (k,k,k, Nf)
        topo_feat = self.encoder(model_input)

        # cube_feat: (N_cubepc, N_f)
        cube_feat = topo_feat[cid[:,0], cid[:,1],cid[:,2]]
        # out_cube: (N_cubepc, 1)
        out_cube = self.decoder_cube.forward_simple(cube_feat)
        if self.predict_type == 'BCE':
            out_cube = out_cube.squeeze(-1)
        return {'pc_cube': self.sigmoid(out_cube)}


    def forward_simple(self, model_input):
        raise NotImplementedError

    def val_cube_pred(self, out_cube, gt_cube):
        out_cube = out_cube.view_as(gt_cube)
        # out_cube = out_cube > 0.5
        # gt_cube type: torch.bool
        # gt_cube = gt_cube > 0.5
        cube_TP = torch.logical_and(out_cube, gt_cube)
        cube_recall = torch.sum(cube_TP) / torch.sum(gt_cube)
        out_cube_sum = max(torch.sum(out_cube), 1)
        cube_prec = torch.sum(cube_TP) / out_cube_sum
        return cube_recall.item(), cube_prec.item()

    def forward_val(self, model_input, gt):
        # k = model_input['info']['grid_size']
        cid = model_input['info']['cube_grid_idx']
        # eid = model_input['info']['edge_grid_idx']

        topo_feat = self.encoder(model_input)
        cube_feat = topo_feat[cid[:,0], cid[:,1],cid[:,2]]
        out_cube = self.decoder_cube.forward_simple(cube_feat)
        if self.predict_type == 'BCE':
            out_cube = self.sigmoid(out_cube) > 0.5
        else:
            out_cube= out_cube[:,0] < out_cube[:,1]
        cube_recall, cube_prec = self.val_cube_pred(out_cube, gt['pc_cube'])

        return {
            'cube_recall': cube_recall,
            'cube_precision': cube_prec
        }


    def predict_curve(self, model_input):
        k = model_input['info']['grid_size']
        cid = model_input['info']['cube_grid_idx']

        topo_feat = self.encoder(model_input)
        cube_feat = topo_feat[cid[:,0], cid[:,1],cid[:,2]]
        out_cube = self.decoder_cube.forward_simple(cube_feat)

        # pred edge idx
        # out_cube: (N_cubepc, 1)
        if self.predict_type == 'BCE':
            out_cube = self.sigmoid(out_cube) > 0.5
            out_cube = out_cube.squeeze(-1)
        else:
            out_cube = out_cube[:,0] < out_cube[:,1]

        # out_cube = (out_cube > 0.5).squeeze(-1)
        peid = cid[out_cube]
        edge_topo = {
            'grid_size': k,
            'cube_idx': peid.detach().cpu().numpy(),
        }

        return edge_topo, peid


class EdgeFaceNet(nn.Module):
    """docstring for EdgeFaceNet"""
    def __init__(self, opt):
        super(EdgeFaceNet, self).__init__()
        # self.k = opt.model.grid_size
        self.encoder = define_grid_encoder(opt.model.encoder)

        self.net_face = CubeFaceNet(opt)

        self.cube_neighbor = torch.LongTensor([
                [-1, 0, 0], [0, -1, 0], [0, 0, -1]
            ])

        self.predict_type = opt.model.predict_type
        if self.predict_type == 'BCE':
            self.sigmoid = nn.Sigmoid()
        elif self.predict_type == 'NLL':
            self.sigmoid = nn.LogSoftmax(dim=1)
        else:
            self.sigmoid = nn.Softmax(dim=1)


    def forward(self, model_input):
        # batch size is 1
        # cid: (N_cubepc, 3); eid: (N_edgeshift, 3)
        k = model_input['info']['grid_size']
        esid = model_input['info']['edge_shift_idx']
        # feat of shape: (k,k,k, Nf)
        topo_feat = self.encoder(model_input)

        edge_feat = topo_feat[esid[:,0], esid[:,1],esid[:,2]]
        # eid_nn: (N_edgeshift, 6, 3)->(N_edgeshift*6, 3)
        N_es = esid.shape[0]
        eid_nn = esid[:,None,:] + self.cube_neighbor
        eid_nn = eid_nn.reshape(N_es*3, 3)
        torch.clamp_(eid_nn, 0, k-1)
        # feat_nn: (N_cubepc, 3, N_f)
        feat_nn = topo_feat[eid_nn[:,0],eid_nn[:,1],eid_nn[:,2]].reshape(N_es,3,-1)
        # out_face: (N_cubepc, 3, 1) / (N_cubepc, 3, 2)
        out_face = self.net_face(edge_feat, feat_nn)
        if self.predict_type == 'BCE':
            out_face = out_face.squeeze(-1)
        else:
            # (N_pc, 2, 3)
            out_face = out_face.transpose(1, 2)
        return {'pc_face': self.sigmoid(out_face)}


    def forward_simple(self, model_input):
        raise NotImplementedError

    def val_face_pred(self, out_face, gt_face):
        out_face = out_face.view_as(gt_face)
        # out_face = out_face > 0.5
        # gt_face = gt_face > 0.5

        check_same = torch.logical_not(torch.logical_xor(out_face, gt_face))
        res = torch.all(check_same, dim=1)
        correct = torch.sum(res) / res.shape[0]
        return correct.item()

    def forward_val(self, model_input, gt):
        k = model_input['info']['grid_size']
        eid = model_input['info']['edge_grid_idx']

        topo_feat = self.encoder(model_input)
        
        N_e = eid.shape[0]
        cube_edge_feat = topo_feat[eid[:,0], eid[:,1],eid[:,2]]
        eid_nn = eid[:,None,:] + self.cube_neighbor
        eid_nn = eid_nn.reshape(N_e*3, 3)
        torch.clamp_(eid_nn, 0, k-1)
        feat_nn = topo_feat[eid_nn[:,0],eid_nn[:,1],eid_nn[:,2]].reshape(N_e,3,-1)

        out_face = self.net_face(cube_edge_feat, feat_nn)
        if self.predict_type == 'BCE':
            out_face = self.sigmoid(out_face) > 0.5
        else:
            out_face = out_face[..., 0] < out_face[..., 1]

        face_correct = self.val_face_pred(out_face, gt['pc_face'])

        return {'face_correct': face_correct}


    def predict_curve(self, model_input, peid):
        k = model_input['info']['grid_size']
        topo_feat = self.encoder(model_input)
        N_e = peid.shape[0]
        if N_e == 0:
            return None

        cube_edge_feat = topo_feat[peid[:,0], peid[:,1],peid[:,2]]
        peid_nn = peid[:,None,:] + self.cube_neighbor
        peid_nn = peid_nn.reshape(N_e*3, 3)
        torch.clamp_(peid_nn, 0, k-1)
        feat_nn = topo_feat[peid_nn[:,0],peid_nn[:,1],peid_nn[:,2]].reshape(N_e,3,-1)

        out_face = self.net_face(cube_edge_feat, feat_nn)
        if self.predict_type == 'BCE':
            out_face = self.sigmoid(out_face) > 0.5
            out_face = out_face.squeeze(-1)
        else:
            out_face = out_face[..., 0] < out_face[..., 1]        

        edge_topo = {
            'cube_faces': out_face.detach().cpu().numpy()
        }

        return edge_topo