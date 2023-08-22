import os, sys, pickle
import torch
import numpy as np
import trimesh

from itertools import product
from scipy.spatial.kdtree import KDTree
from torch.utils.data import Dataset


# def check_name(name_encoder, name_output):
#     encoders = ['KNNPointcloud']
#     outputs = ['NerVEGrid']
#     if name_encoder not in encoders or name_output not in outputs:
#         print('Wrong dataset name, should be form encoder_output')
#         print('Encoders: {}, output:{}'.format(encoders, outputs))
#         raise NameError


class RawPCDataset():
    """
    Data format
    """

    def __init__(self, params):
        super().__init__()
        p = params
        self.k = p['grid_size']
        self.step = 2. / self.k
        self.cube_shift_mode = p['cube_shift_mode']
        self.knn_size = 8

        self.pc_file = p['pc_file']
        self.dataset_path = p['data_path']
        self.names = np.loadtxt(p['file_list'], dtype=str)


    def __len__(self):
        return len(self.names)


    def pc_normalize(self, pc):
        # normalize_factor be consistent with dataset
        normalize_factor = 0.9
        bmin,bmax = np.min(pc, axis=0), np.max(pc, axis=0)
        center, scale = (bmin+bmax)/2, np.max(bmax-bmin)/2
        pc = (pc - center)*(normalize_factor / scale)
        return pc

    def KNN_idx(self, pc):
        tree = KDTree(pc, leafsize=self.knn_size)
        _, idx = tree.query(pc, k=self.knn_size)
        return idx

    def process_input(self, pc_path):
        name, ext = os.path.splitext(os.path.basename(pc_path))
        if ext == '.npy':
            pc = np.load(pc_path)
        elif ext == '.ply':
            pcmesh = trimesh.load(pc_path, process=False)
            pc = np.asarray(pcmesh.vertices)
        elif ext == '.pkl':
            pc_data = pickle.load(open(pc_path, 'rb'))
            pc = pc_data['pc']
        else:
            raise NotImplementedError('Not supported ext, now can process: npy,ply,pkl(key:points)')
    
        pc = self.pc_normalize(pc)
        
        # pre-calculate grid info of pc
        k = self.k
        step = 2./k
        pcid = np.floor((pc + 1) / step).astype(int)
        grid = np.zeros((k,k,k), dtype=bool)
        grid[pcid[:,0], pcid[:,1], pcid[:,2]] = True

        # shift to neighbor(26/27) or 6/27
        if self.cube_shift_mode == 'full':
            ijl = product(range(3), range(3), range(3))
        elif self.cube_shift_mode == 'six':
            ijl = [[1,1,1], [0,1,1],[2,1,1], [1,0,1],[1,2,1], [1,1,0], [1,1,2]]
        else:
            ijl = None
            # print('No shift for input point cloud')

        if ijl is not None:
            tmp_mask = np.copy(grid[1:-1,1:-1,1:-1])
            for i,j,l in ijl:
                grid[i:k-2+i, j:k-2+j, l:k-2+l] = np.logical_or(tmp_mask, grid[i:k-2+i, j:k-2+j, l:k-2+l])

        cid = np.argwhere(grid)
        knn_idx = self.KNN_idx(pc)
        return {
            'pc': pc,
            'pc_KNN_idx': knn_idx,
            'pc_grid_idx': pcid,
            'cube_grid_idx': cid,
        }
    
    def network_input_normalize(self, encoder_input, normalize_mode):
        pc = encoder_input['pc']
        pc_KNN_idx = encoder_input['pc_KNN_idx']
        
        if normalize_mode == 'cube_face':
            # ##version1: substract center and multiply reso
            # centralize each cluster, and scale to cube level
            pos = pc[pc_KNN_idx]
            pos -= pc.reshape(-1,1,3)
            pos *= self.k
        elif normalize_mode == 'geom':
            # ###version2: local cube coordinates
            # convert to local cube coordiantes 
            pc_cube_idx = encoder_input['pc_grid_idx']
            pos = pc[pc_KNN_idx]
            centers = self.step*pc_cube_idx + (self.step/2. - 1)
            pos -= centers.reshape(-1,1,3)
            pos *= self.k
        else:
            raise NotImplementedError
        
        return {
            'pc_KNN_pos': torch.from_numpy(pos).float()
        }

    def get_data(self, idx, normalize):
        fname = self.names[idx]
        data_path = os.path.join(self.dataset_path, fname, self.pc_file)
        # data_path = os.path.join(self.dataset_path, 'pc_norm_0.9', fname)

        encoder_input = self.process_input(data_path)
        model_input = self.network_input_normalize(encoder_input, normalize)

        info = {
            'name': fname,
            'grid_size': self.k,
            'file_path': data_path,
            'points': torch.from_numpy(encoder_input['pc']).float(),
            'pc_KNN_idx': encoder_input['pc_KNN_idx'],
            'pc_grid_idx': encoder_input['pc_grid_idx'],
            'cube_grid_idx': torch.from_numpy(encoder_input['cube_grid_idx']).long(),
        }
        
        return model_input, info


class EdgeDataset(Dataset):
    """
    Data format
    """
    def __init__(self, params):
        super().__init__()
        p = params
        self.dataset_path = p['root']
        self.output_element = p['output_element']

        encoder_type, output_type = p['encoder_type'], p['output_type']
        # check_name(encoder_type, output_type)
        this_file = sys.modules[__name__]
        DataEncoder = getattr(this_file, f'{encoder_type}_Base')
        DataOutput = getattr(this_file, f'Base_{output_type}')
        self.data_encoder = DataEncoder(p)
        self.data_output = DataOutput(p)
        
        self.mode = p['mode']
        if self.mode in ['train', 'val']:
            self.names = np.loadtxt(os.path.join(self.dataset_path, f'{self.mode}.txt'), dtype=int)
        elif self.mode == 'test':
            val_names = np.loadtxt(os.path.join(self.dataset_path, 'val.txt'), dtype=int)
            test_names = np.loadtxt(os.path.join(self.dataset_path, 'test.txt'), dtype=int)
            self.names = np.concatenate([val_names, test_names])
        else:
            raise NotImplementedError('wrong dataset mode')


    def collate_fn(self, batch):
        if len(batch) > 1:
            raise ValueError('Batch size can only be 1')

        return batch[0]

    def locate_cube(self, points):
        k = self.data_encoder.k
        step = 2./k
        return np.floor((points + 1) / step).astype(int)

    def cube_shift(self, k, cidx):
        grid = np.zeros((k,k,k), dtype=bool)
        grid[cidx[:,0], cidx[:,1], cidx[:,2]] = True

        ijl = [[1,1,1], [0,1,1],[2,1,1], [1,0,1],[1,2,1], [1,1,0], [1,1,2]]
        tmp_mask = np.copy(grid[1:-1,1:-1,1:-1])
        for i,j,l in ijl:
            grid[i:k-2+i, j:k-2+j, l:k-2+l] = np.logical_or(tmp_mask, grid[i:k-2+i, j:k-2+j, l:k-2+l])

        return np.argwhere(grid)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        info = {}
        fname = '%08d' % int(self.names[idx])
        item_path = os.path.join(self.dataset_path, fname)

        encoder_input = self.data_encoder.encoder_get_item(item_path)

        func_get_data = getattr(self.data_output, f'{self.mode}_get_item')
        gt = func_get_data(item_path)

        if isinstance(self.data_encoder, KNNPointcloud_Base):
            model_input = {
                'pc_KNN_pos': encoder_input['pc_KNN_pos'],
            }
        else:
            raise NotImplementedError

        cgid = encoder_input['cube_grid_idx']
        eid = gt['edge_grid_idx']
        info.update({
            'name': self.names[idx],
            'item_path': item_path,
            'grid_size': self.data_encoder.k,
            'points': encoder_input['points'],
            'cube_grid_idx': cgid,
            'edge_grid_idx': eid,
        })

        if self.output_element == 'cube':
            # shape, pc_cube: (N_cubepc,), pc_face: (N_cubepc, 3)
            pc_cube = gt['grid_cube'][cgid[:,0], cgid[:,1], cgid[:,2]]
            if self.mode == 'train':
                # for computing BCE loss, or it will be bool
                # pc_cube = pc_cube.long()
                pc_cube = pc_cube.float()

            output_gt = {'pc_cube': pc_cube}

        elif self.output_element == 'face':
            if self.mode == 'train':
                esid = gt['edge_shift_idx']
                info['edge_shift_idx'] = esid
                pc_face = gt['grid_face'][esid[:,0], esid[:,1], esid[:,2]]
                # pc_face = pc_face.long()
                pc_face = pc_face.float()
            else:
                pc_face = gt['grid_face'][eid[:,0], eid[:,1], eid[:,2]]

            output_gt = {'pc_face': pc_face}

        elif self.output_element == 'geom':
            output_gt = {
                'pc_point': gt['grid_points'],
            }
        elif self.output_element == 'gen_curve':
            output_gt = {}
        else:
            raise NotImplementedError('Not supported element')

        return model_input, output_gt, info


class Base_NerVEGrid():
    """docstring for Base_NerVEGrid"""
    def __init__(self, params):
        self.cube_file = params['cube_file']

    def data_get_item(self, item_path):
        cube_file_path = os.path.join(item_path, self.cube_file)
        with open(cube_file_path, 'rb') as f:
            cube_data = pickle.load(f)

        k = cube_data['grid_size']
        edge_grid_idx = cube_data['cube_idx']
        ekid = edge_grid_idx @ np.asarray([1,k,k**2])
        edge_dict = {kid: num for num,kid in enumerate(ekid)}
        
        points = cube_data['cube_points']
        step = 2. / k

        # cube; faces;
        eg = edge_grid_idx
        cubes = np.zeros((k,k,k), dtype=bool)
        faces = np.zeros((k,k,k,3), dtype=bool)
        cubes[eg[:,0], eg[:,1], eg[:,2]] = True
        faces[eg[:,0], eg[:,1], eg[:,2]] = cube_data['cube_faces']
        
        # transform points to local cube coordinates
        centers = step*edge_grid_idx + (step/2. - 1.)
        points -= centers
        points *= k

        return {
            'cubes': cubes,
            'faces': faces,
            'points': points,
            'edge_grid_idx': edge_grid_idx,
            'cid_dict': edge_dict,
            'eid_faces': cube_data['cube_faces']
        }

    def train_get_item(self, item_path):
        data = self.data_get_item(item_path)
        # one-neighbor shift to edge_grid_idx, for face and pair training
        eid = data['edge_grid_idx']
        k = data['cubes'].shape[0]
        grid = np.zeros((k,k,k), dtype=bool)
        grid[eid[:,0], eid[:,1], eid[:,2]] = True

        # shift to neighbor(26/27)
        ijl = product(range(3), range(3), range(3))
        tmp_mask = np.copy(grid[1:-1,1:-1,1:-1])
        for i,j,l in ijl:
            grid[i:k-2+i, j:k-2+j, l:k-2+l] = np.logical_or(tmp_mask, grid[i:k-2+i, j:k-2+j, l:k-2+l])

        eid_shift = np.argwhere(grid)
        return {
            'grid_cube' : torch.from_numpy(data['cubes']),
            'grid_face': torch.from_numpy(data['faces']),
            'grid_points': torch.from_numpy(data['points']).float(),
            'edge_grid_idx': torch.from_numpy(data['edge_grid_idx']).long(),
            'edge_shift_idx': torch.from_numpy(eid_shift).long(),
            'eid_faces': data['eid_faces']
        }

    def test_get_item(self, item_path):
        data = self.data_get_item(item_path)
        return {
            'grid_cube' : torch.from_numpy(data['cubes']),
            'grid_face': torch.from_numpy(data['faces']),
            'grid_points': torch.from_numpy(data['points']).float(),
            'edge_grid_idx': torch.from_numpy(data['edge_grid_idx']).long(),
            'eid_faces': data['eid_faces']
        }

    def val_get_item(self, item_path):
        return self.test_get_item(item_path)
        

class KNNPointcloud_Base():
    """
    Data format

    """
    def __init__(self, params):
        self.encoder_file = params['encoder_file']
        # self.leaf_size = params['leaf_size']
        self.k = params['grid_size']
        self.step = 2./ self.k
        self.normalize_mode = params['pc_normalize']


    def encoder_get_item(self, item_path):
        encoder_file_path = os.path.join(item_path, self.encoder_file)
        with open(encoder_file_path, 'rb') as f:
            pc_data = pickle.load(f)

        pc = pc_data['pc']
        pc_KNN_idx = pc_data['pc_KNN_idx']
        cube_grid_idx = pc_data['cube_grid_idx']
        
        if self.normalize_mode == 'topo':
            # ##version1: substract center and multiply reso
            # centralize each cluster, and scale to cube level
            pos = pc[pc_KNN_idx]
            pos -= pc.reshape(-1,1,3)
            pos *= self.k
        elif self.normalize_mode == 'geom':
            # ###version2: local cube coordinates
            # convert to local cube coordiantes 
            pc_cube_idx = pc_data['pc_grid_idx']
            pos = pc[pc_KNN_idx]
            centers = self.step*pc_cube_idx + (self.step/2. - 1)
            pos -= centers.reshape(-1,1,3)
            pos *= self.k
        elif self.normalize_mode == 'pc':
            pc_cube_idx = pc_data['pc_grid_idx']
            centers = self.step*pc_cube_idx + (self.step/2. - 1)
            pos = pc - centers
            pos *= self.k
        else:
            raise NotImplementedError('Not recognized mode')

        return {
            'points': torch.from_numpy(pc).float(),
            'pc_KNN_pos': torch.from_numpy(pos).float(),
            'cube_grid_idx': torch.from_numpy(cube_grid_idx).long(),
        }
