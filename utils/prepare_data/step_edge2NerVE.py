import os, glob, pickle
import numpy as np
from time import time
from itertools import product

class NerVEGrid():
    """docstring for NerVEGrid"""
    def __init__(self, grid_size, offset_config):
        self.grid_size = grid_size
        k = grid_size

        self.pnt_faces = np.zeros((k,k,k, 3), dtype=bool)
        self.pnt_cubes = np.zeros((k,k,k), dtype=bool)

        self.step = 2./ k
        self.offset_config = offset_config
        if 'offset' in offset_config:
            self.stable_offset = offset_config['offset']
            self.given_offset = True
        else:
            self.stable_offset = 0.
            self.given_offset = False

        self.corner = -np.ones(3)
        self.k_base = np.asarray([1, k, k**2])

    def init(self):
        self.pnt_faces.fill(False)
        self.pnt_cubes.fill(False)

    def locate_cube(self, point):
        return np.floor((point - self.corner) / self.step).astype(int)

    def random_offset(self, points):
        # assume points in [-1,1]^3
        # rescale to [0,k]^3, k is grid size
        t0 = time()
        num_try = self.offset_config['num_try']
        thres = self.offset_config['thres']
        # points = (points + 1) *(self.grid_size / 2)
        # temp = np.abs(points - np.around(points))
        # print(f'Before: Mean (16x)dist: {np.mean(temp)}, Num < {thres}: {np.sum(temp < thres)}')

        offsets = np.random.uniform(-0.49, 0.49, size=(num_try, 3))
        res = []
        for offset in offsets:
            temp = points + offset
            temp = np.abs(temp - np.around(temp))
            dist = np.mean(temp)
            num = np.sum(temp < thres)
            res.append([dist, num])

        res = np.asarray(res)
        num_idx = np.argsort(res[:,1])[:5]
        # dist_idx = np.argsort(res[:,0])[-10:]
        # print('dist ranking max 10: \n', res[dist_idx, :])
        # print('num ranking min 10: \n', res[num_idx, :])
        minmax = np.argmax(res[num_idx, 0])
        best_idx = num_idx[minmax]
        # print(f'Done, time:{time()-t0}, choose offset with dist:{res[best_idx,0]}, num<{thres}: {res[best_idx,1]}')
        self.stable_offset = offsets[best_idx] / (self.grid_size / 2)


    def cube_segment_intersect(self, pts, cube_idx, seg_idx, face_intersects):
        # given segment, find two cube of vertices, 
        # find out intersected point, face, tangent(normalized segment), saved in face_intersects
        int_points = []
        int_neighbor = []
        for idx in seg_idx:
            next_idx = (idx + 1) % pts.shape[0]
            cube1, cube2 = cube_idx[idx], cube_idx[next_idx]
            p1, p2 = pts[idx], pts[next_idx]

            # diff:(n,1); planes:(n,1)
            diff = np.argwhere(cube1 != cube2)
            planes = self.step* np.max([cube1, cube2], axis=0)[diff] - 1

            coefs = np.abs(p1[diff] - planes) / np.abs(p1[diff] - p2[diff])
            coefs = np.sort(coefs, axis=0)
            p1p2 = p2 - p1
            pts_int = p1 + coefs* p1p2
            # temp for DC
            tan_int = p1p2 / np. linalg.norm(p1p2)
            tan_int = np.tile(tan_int, (pts_int.shape[0], 1))
            int_points.append(np.concatenate([pts_int, tan_int], axis=1))
            int_neighbor.extend([idx]*pts_int.shape[0])
            
            # label extra face
            if diff.shape[0] == 2:
                mid_pts = np.mean(pts_int, axis=0)
                i,j,k = self.locate_cube(mid_pts)
                self.pnt_cubes[i,j,k] = True

            if diff.shape[0] == 3:
                f1,f2,f3 = np.argsort(coefs.flatten())
                mid1 = (pts_int[f1] + pts_int[f2]) / 2
                i,j,k = self.locate_cube(mid1)
                self.pnt_cubes[i,j,k] = True
                mid2 = (pts_int[f2] + pts_int[f3]) / 2
                i,j,k = self.locate_cube(mid2)
                self.pnt_cubes[i,j,k] = True

            # find face for pts_int 
            int_cube_idx = self.locate_cube(pts_int)
            corners = self.step* int_cube_idx - 1
            flags = np.abs(corners - pts_int) < 1e-12

            # face_id = np.argmax(flags, axis=1)
            faces_id = np.argwhere(flags)
            n_ints = pts_int.shape[0]
            for face_id in faces_id:
                num, fid = face_id
                i,j,k = int_cube_idx[num]
                # data = np.concatenate([pts_int[num], tan_int])
                face_str = f'{i}_{j}_{k}_{fid}'
                face_intersects[face_str] = True

        # Use intersections to calculate points in cubes(and tangents)
        int_points = np.vstack(int_points)
        int_pts = int_points[:,:3]
        n_int = int_pts.shape[0]
        int_neighbor = np.asarray(int_neighbor)
        p0s, p1s = int_pts[:-1], int_pts[1:]
        p0n = int_neighbor[:-1] + 1
        p1n = int_neighbor[1:]
        segs0 = np.linalg.norm(pts[p0n] - p0s, axis=1)
        segs1 = np.linalg.norm(pts[p1n] - p1s, axis=1)
        segs = (segs1 - segs0) / 2.
        
        mid_ptstan = np.zeros((n_int-1, 6))
        for i in range(n_int-1):
            if (p0n[i]+p1n[i]) % 2 == 0:
                pid = (p0n[i]+p1n[i]) // 2
                pid0, pid1 = pid-1, pid+1
                if segs[i] >= 0:
                    tan = pts[pid1] - pts[pid]
                    tan /= np.linalg.norm(tan)
                    mid_ptstan[i,3:] = tan
                    mid_ptstan[i,:3] = pts[pid] + abs(segs[i])*tan
                else:
                    tan = pts[pid0] - pts[pid]
                    tan /= np.linalg.norm(tan)
                    mid_ptstan[i,3:] = -tan
                    mid_ptstan[i,:3] = pts[pid] + abs(segs[i])*tan
            else:
                pid0 = int((p0n[i]+p1n[i])/2)
                pid1 = pid0 + 1
                pos = (pts[pid0] + pts[pid1]) / 2.
                tan = pts[pid1] - pts[pid0]
                tan /= np.linalg.norm(tan)
                pos += segs[i]*tan
                mid_ptstan[i,:3] = pos
                mid_ptstan[i,3:] = tan

        return int_points, mid_ptstan


    def neighbor_cubes(self, point):
        # find all cubes covering the given point
        idx = self.locate_cube(point)
        corner = self.step* idx - 1.

        flag = np.abs(point - corner) < 1e-12
        # , , 
        num_zero = np.sum(flag)
        if num_zero == 0:
            return np.asarray([idx @ self.k_base])

        # 1 True: lie on face
        if num_zero == 1:
            fid = np.argwhere(flag).flatten()[0]
            offset = np.zeros((2,3), dtype=int)
            offset[1, fid] = 1

            return (idx - offset) @ self.k_base

        # 2 True: lie on edge
        if num_zero == 2:
            i, j = np.argwhere(flag).flatten()
            offset = np.zeros((4,3), dtype=int)
            offset[1, i] = 1
            offset[2, j] = 1
            offset[3, i] = 1
            offset[3, j] = 1

            return (idx - offset) @ self.k_base

        # 3 True: exact the corner
        if num_zero == 3:
            offset = np.asarray(list(product([0,1], [0,1], [0,1])), dtype=int)
            return (idx - offset) @ self.k_base


    def get_face_idx(self, cube_idx, face_point):
        v_cidx = self.locate_cube(face_point)

        corner = self.step* v_cidx - 1
        flag = np.abs(corner - face_point) < 1e-12
        flag_sum = np.sum(flag)
        if flag_sum != 1:
            raise ValueError('flag sum !=1. Point lie on edge or vertex')

        diff = np.abs(cube_idx - v_cidx) > 0
        diff_sum = np.sum(diff)
        if diff_sum > 1:
            raise ValueError('diff sum !=1. Point lie on edge or vertex')

        if diff_sum == 0:
            # point on this cube
            return np.argmax(flag)
        if diff_sum == 1:
            # point on adjacent cube
            return np.argmax(flag)+3


    def filter_noclosed(self, verts, verts_closed):
        vclosed = np.asarray([v for v,vid in verts_closed])
        vids = np.asarray([vid for v,vid in verts_closed])

        verts_noclosed = []
        for vert in verts:
            vid = vert @ self.k_base
            if vid in vids:
                dist = np.linalg.norm(vclosed-vert, axis=1)
                if np.any(dist < 1e-15):
                    # vert be a vertex in a closed curve
                    continue
            else:
                verts_noclosed.append(vert)

        return np.asarray(verts_noclosed)


    def solve_QEF(self, pts, tan):
        n_pts = pts.shape[0]
        A = np.eye(3) - (tan.T @ tan) / (2*n_pts)
        pw_ip = np.einsum('ij,ij->i', pts, tan)
        term = np.einsum('ij,i->j', tan, pw_ip)
        b = np.mean(pts, axis=0) - term / (2*n_pts)

        res = np.linalg.solve(A, b)
        return res

    def calc_cube_attr(self, intersect_pts, N_cube, idx_dict):
        use_DC_point = False
        cubes_pts = np.zeros((N_cube, 6))

        cube_ptstan = {}
        for key, val in intersect_pts.items():
            # pts = val['int_pts']
            int_data = val['int_pts']
            pts = int_data[:,:3]
            tan = int_data[:,3:]
            if pts.shape[0] < 2:
                # corners are adjacent or overlapped
                continue

            mid_ptstan = val['mid_ptstan']
            mid_cube_idx = self.locate_cube(mid_ptstan[:,:3])
            mc = mid_cube_idx
            kidx = mc @ self.k_base

            for i in range(mc.shape[0]):
                kid = idx_dict[kidx[i]]

                final_point = mid_ptstan[i]
                if use_DC_point:
                    final_point = np.zeros(6)
                    final_point[:3] = self.solve_QEF(pts[i:i+2], tan[i:i+2])

                if kid in cube_ptstan:
                    cube_ptstan[kid].append(final_point)
                else:
                    cube_ptstan[kid] = [final_point]

        for kid,val in cube_ptstan.items():
            cubes_pts[kid] = np.mean(val, axis=0)

        return cubes_pts


    def load_step_edges(self, data_path):
        """
        content:
        vertices : [v0, ..., vm]
        edges:{edge0, ..., edgeN}
        edge: 
          -is_closed: bool
          -type: type of edge(Line,Circle,BSpline)
          -parameters: parameters of samples
          -samples: position of samples
        """
        # name, ext = os.path.splitext(os.path.basename(data_path))
        dirname = os.path.dirname(data_path)
        output_path = os.path.join(dirname, f'nerve_reso{self.grid_size}.pkl')
        if os.path.exists(output_path):
            return

        # print('Loaded ', os.path.basename(dirname))
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        t0 = time()
        self.init()

        verts = np.asarray(data['vertices'])
        if not self.given_offset:
            if not self.offset_config['vertex_only']:
                points = [verts]
                for key in data:
                    if 'edge' not in key:
                        continue
                    edge_data = data[key]
                    pts = np.asarray(edge_data['samples'])
                    points.append(pts)

                points = np.concatenate(points, axis=0)
            else:
                points = verts

            self.random_offset(points)
        # for numerical stability concern
        verts += self.stable_offset

        face_intersects = {}
        intersect_pts = {}
        verts_closed = []
        for key in data:
            if 'edge' not in key:
                continue

            edge_data = data[key]
            pts = np.asarray(edge_data['samples'])
            # for numerical stability concern
            pts += self.stable_offset
            cube_idx = self.locate_cube(pts)

            self.pnt_cubes[cube_idx[:,0], cube_idx[:,1], cube_idx[:,2]] = True

            # consider all line-segments(points are consecutive on curve edge)
            # cube_idx to unique 1d idx
            start_idx = cube_idx @ self.k_base

            next_idx = np.roll(start_idx, -1)
            if not edge_data['is_closed']:
                next_idx[-1] = start_idx[-1]

            seg_idx = np.nonzero(start_idx != next_idx)[0]
            if seg_idx.shape[0] == 0:
                continue

            int_points, mid_ptstan = self.cube_segment_intersect(
                pts, cube_idx, seg_idx, face_intersects)

            intersect_pts[key] = {
                'int_pts': int_points, 
                'mid_ptstan': mid_ptstan
            }
            # make face pair for corner of closed curve 
            if edge_data['is_closed']:
                verts_closed.append([pts[0], start_idx[0]])
                    
        for face_str in face_intersects.keys():
            i,j,k,l = list(map(int, face_str.split('_')))
            self.pnt_faces[i,j,k,l] = True

        # # Compute GT point in all marked cubes, 
        cubes_idx = np.argwhere(self.pnt_cubes)
        N_cube = cubes_idx.shape[0]
        k_idx = cubes_idx @ self.k_base
        dict_cube_idx = {idx: i for i,idx in enumerate(k_idx)}

        # compute points and tangents inside the cubes
        cubes_pts = self.calc_cube_attr(intersect_pts, N_cube, dict_cube_idx)

        verts_noclosed = self.filter_noclosed(verts, verts_closed)
        # replace some with verts, check difference with verts
        for vert in verts_noclosed:
            # all cubes covering vert
            v_cid = self.neighbor_cubes(vert)
            for cid in v_cid:
                if cid in dict_cube_idx:
                    vid = dict_cube_idx[cid]
                    cubes_pts[vid, :3] = vert

        cubes_faces = self.pnt_faces[cubes_idx[:,0], cubes_idx[:,1], cubes_idx[:,2]]
        res = {
            'grid_size': self.grid_size,
            'cube_idx': cubes_idx,
            'cube_points': cubes_pts[:,:3],
            'cube_tan': cubes_pts[:,3:],
            'cube_faces': cubes_faces,
            'stable_offset': self.stable_offset
        }

        with open(output_path, 'wb') as f:
            pickle.dump(res, f)
        
        out_curve_path = os.path.join(dirname, f'nerve_reso{self.grid_size}_curve.pkl')
        self.output_curve(self.grid_size, cubes_idx, 
                        cubes_pts[:,:3], cubes_faces, out_curve_path)
        # print('Done, time cost: ', time()-t0)


    def output_curve(self, k, cube_idx, cube_pts, cube_faces, output_path):
        k_base = np.asarray([1, k, k**2])
        k_idx = cube_idx @ k_base
        dict_cube_idx = {idx: i for i,idx in enumerate(k_idx)}

        edges = []
        for idx, face_status in zip(cube_idx, cube_faces):
            cid = idx @ k_base
            for num in range(3):
                if not face_status[num]:
                    continue
                kidx = idx.copy()
                kidx[num] -= 1
                kidx = kidx @ k_base
                if kidx not in dict_cube_idx:
                    continue

                edges.append([dict_cube_idx[cid], dict_cube_idx[kidx]])

        curve = {
            'points': cube_pts,
            'edges': edges,
            'stable_offset': self.stable_offset
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(curve, f)


def dataset_process():
    dataset_path = '/Path/to/your/ABC/NerVE64Dataset'
    offset_config = {
        'vertex_only': True,
        'num_try': 100,
        'thres': 0.05,
    }
    reso = 64
    failed_list = []
    t0 = time()
    iters = glob.glob(os.path.join(dataset_path, f'**/step_edge_reso{reso}.pkl'), recursive=True)
    for k, data_path in enumerate(iters):
        if k % 200 == 0:
            print(f'Process {k}, time elapsed: {time()-t0}')

        nerve_grid = NerVEGrid(reso, offset_config)
        try:
            nerve_grid.load_step_edges(data_path)
        except Exception as e:
            print(f'Failed: {data_path}')
            failed_list.append(data_path)
    
    print(f'Num of failed: {len(failed_list)}')
    failed_path = os.path.join(dataset_path, 'failed.txt')
    np.savetxt(failed_path, failed_list, fmt='%s')
    print('Total time cost: ', time()-t0)

if __name__ == '__main__':
    dataset_process()
