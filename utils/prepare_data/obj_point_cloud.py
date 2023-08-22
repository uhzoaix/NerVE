import os, pickle
import numpy as np
import trimesh
from time import time
from itertools import product
from scipy.spatial import KDTree


def points_normalize(pc, factor, offset):
    bmin, bmax = np.min(pc, axis=0), np.max(pc, axis=0)
    center, scale = (bmin+bmax)/2., np.max(bmax-bmin)/2.
    pc -= center
    pc *= factor / scale
    pc += offset

    return pc


def precompute_index(pc, k, cube_shift=True):
    step = 2./k
    pcid = np.floor((pc + 1) / step).astype(int)
    grid = np.zeros((k,k,k), dtype=bool)
    grid[pcid[:,0], pcid[:,1], pcid[:,2]] = True

    if cube_shift:
        # shift to neighbor(26/27)
        ijl = product(range(3), range(3), range(3))
        tmp_mask = np.copy(grid[1:-1,1:-1,1:-1])
        for i,j,l in ijl:
            grid[i:k-2+i, j:k-2+j, l:k-2+l] = np.logical_or(tmp_mask, grid[i:k-2+i, j:k-2+j, l:k-2+l])

    cid = np.argwhere(grid)
    return pcid, cid

def KNN_idx(pc, leafsize):
    tree = KDTree(pc, leafsize=leafsize)
    _, idx = tree.query(pc, k=leafsize)
    return idx


def get_offset(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data['stable_offset']

def output_pc_data(pc, k, output_path, offset=None, leaf_size=8):
    pcid, cid = precompute_index(pc, k)
    pc_KNN_idx = KNN_idx(pc, leafsize=leaf_size)

    res = {
        'pc': pc,
        'pc_grid_idx': pcid,
        'pc_KNN_idx': pc_KNN_idx,
        'cube_grid_idx': cid,
        'grid_size': k,
        'leaf_size': leaf_size,
        'stable_offset': offset
    }

    with open(output_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    root_path = '/home/uhzoaix/Work/data/ABC'
    mesh_path = os.path.join(root_path, 'abc_0000_obj_v00')
    file_path = os.path.join(root_path, 'NerVE64Dataset', 'all.txt')
    output_path = os.path.join(root_path, 'NerVE64Dataset')

    k = 64
    leaf_size = 8
    # upper bound for number of points
    ub_numpc = 50000

    file_list = np.loadtxt(file_path, dtype=int)

    t0 = time()
    for count, idx in enumerate(file_list):
        if count % 100 == 0:
            print(f'Process: {idx}')

        fname = '%08d' % idx
        output_fpath = os.path.join(output_path, fname)
        os.makedirs(output_fpath, exist_ok=True)
        out_file = os.path.join(output_fpath, f'pc_obj.pkl')

        if os.path.exists(out_file):
            print('Existed: ', out_file)
            continue
        
        # use nerve data only to get the offset
        nerve_path = os.path.join(output_fpath, f'nerve_reso{k}.pkl')
        fpath = os.path.join(mesh_path, fname)
        if len(os.listdir(fpath)) == 0:
            print(f'### No obj file for {idx}')
            continue
        obj_name = os.listdir(fpath)[0]
        mesh = trimesh.load(os.path.join(fpath, obj_name), process=False)
        num_pc = mesh.vertices.shape[0]
        if num_pc <= ub_numpc:
            pc = np.asarray(mesh.vertices)
        else:
            pc = mesh.sample(ub_numpc)
            
        stable_offset = get_offset(nerve_path)
        pc = points_normalize(pc, factor=0.9, offset=stable_offset)

        output_pc_data(pc, k, out_file, 
                       stable_offset, leaf_size=leaf_size)


    print('Done, time cost: ', time()-t0)
