import os, pickle
import numpy as np
import trimesh
import scipy.interpolate as spi
import os.path as op
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from copy import deepcopy
from itertools import combinations
from time import time
import timeout_decorator
from eval_cad_curve import convert_cad_to_pwl

timeout_seconds = 5

class PWLCurve():
    """
    Piece-wise linear curve
    """
    def __init__(self, curve_data_path) -> None:
        with open(curve_data_path, 'rb') as f:
            curve_data = pickle.load(f)

        self.points = curve_data['points']
        self.edges = curve_data['edges']
        self.k = 64
        self.k_base = np.asarray([1,self.k,self.k**2])

        # prepare for geomtry and topology analysis
        self.tree = KDTree(self.points)
        self.vv = self.__calc_neighbor()
        self.degree = np.asarray([len(nv) for nv in self.vv])

        # grid cube
        self.step = 2./ self.k
        self.points_idx = ((self.points + 1) / self.step).astype(int)

        # previous choice: radius: 3*step; delete length: 4
        self.delete_points = np.zeros(self.points.shape[0], dtype=bool)
        self.add_edge = []
        self.temp_degree = self.degree.copy()
        self.is_clean = False

        # for spline fitting 
        self.line_thres = 1e-3
        self.circle_thres = 1e-3


    def set_parameters(self, param):
        # for cleaning
        self.cossim_thres = np.sqrt(2)
        self.query_ball_radius = param['query_ball_radius']*self.step
        self.extend_max_count = param['extend_max_count']
        self.delete_max_length = param['delete_max_length']
        self.only_extend = param['only_extend']
        self.closed_path_thres = param['closed_path_dist']*self.step


    @timeout_decorator.timeout(timeout_seconds)
    def curve_cleaning(self):
        # find all 1-degree points connect them with nearest 1-degree
        vid_d1 = np.argwhere(self.degree == 1)
        temp_vv = deepcopy(self.vv)
        for vid in vid_d1:
            vid = vid[0]
            if self.temp_degree[vid] != 1:
                continue
            # print(f'Processing {vid}, {self.points[vid]}')
            vid_n = self.vv[vid][0]
            ban_list = [vid_n]
            ban_list.extend(self.vv[vid_n])
            pid_ball = self.tree.query_ball_point(self.points[vid], r=self.query_ball_radius)
            # print(f'query ball pid: {pid_ball}, degree:{self.degree[pid_ball]}')
            pid_ball = [pid for pid in pid_ball if pid not in ban_list and self.temp_degree[pid] == 1]

            if len(pid_ball) > 0:
                # dist = np.linalg.norm(self.points[pid_ball]-self.points[vid], axis=1)
                # new_vid = pid_ball[np.argmin(dist)]
                # calculate tangent change
                tan_vid = self.points[vid] - self.points[vid_n]
                tan_vid = np.tile(tan_vid, (len(pid_ball), 1))

                tan_add = self.points[pid_ball] - self.points[vid]
                pidball_n = [self.vv[idx][0] for idx in pid_ball]
                tan_pidball = self.points[pidball_n] - self.points[pid_ball]
                cossim1 = self.batch_cos_similarity(tan_vid, tan_add)
                cossim2 = self.batch_cos_similarity(tan_add, tan_pidball)
                cossim = cossim1 + cossim2
                if np.max(cossim) > self.cossim_thres:
                    new_vid = pid_ball[np.argmax(cossim)]
                    self.add_edge.append([vid, new_vid])
                    self.temp_degree[vid] = 2
                    self.temp_degree[new_vid] = 2
                    temp_vv[vid].append(new_vid)
                    temp_vv[new_vid].append(vid)
                    # print(f'connect {vid} {new_vid}')
            
        # delete small(< delete_max_length) isolated edges 
        vid_d1 = [vid[0] for vid in vid_d1 if self.temp_degree[vid[0]] == 1]
        while(len(vid_d1) > 0):
            vid = vid_d1.pop()
            # print(f'Process {vid}, {self.points[vid]}')
            if self.temp_degree[vid] == 1:
                chain = self.get_chain(vid, temp_vv)
                # print(f'length of chain:{len(chain)}, last vertex:{chain[-1]}, deg:{self.temp_degree[chain[-1]]}')
                if len(chain) < self.delete_max_length:
                    self.delete_points[chain[:-1]] = True
                    self.temp_degree[chain[:-1]] = 0
                    for idx in chain[:-1]:
                        temp_vv[idx] = []
                    self.temp_degree[chain[-1]] -= 1
                    # update temp_vv for last chain vertex if necessary
                    temp_vv[chain[-1]].remove(chain[-2])

        # try to extend d1 vertex
        vid_d1 = [idx[0] for idx in np.argwhere(self.temp_degree == 1)]
        while (len(vid_d1) > 0):
            vid = vid_d1.pop()
            # print(f'Processing {vid}, degree:{self.temp_degree[vid]}, vv:{temp_vv[vid]}')
            if self.temp_degree[vid] != 1:
                continue
            vid_n = temp_vv[vid][0]
            ban_list = [vid_n]
            ban_list.extend(temp_vv[vid_n])
            # next vid
            nvid = vid
            for count in range(self.extend_max_count):
                pid_ball = self.tree.query_ball_point(self.points[nvid], r=0.6*self.query_ball_radius)
                pid_ball = [idx for idx in pid_ball if idx not in ban_list]
                if len(pid_ball) == 0:
                    break
                tan_vid = self.points[nvid] - self.points[vid_n]
                tan_vid = np.tile(tan_vid, (len(pid_ball), 1))
                tan_add = self.points[pid_ball] - self.points[nvid]
                cossim = self.batch_cos_similarity(tan_vid, tan_add)
                new_vid = pid_ball[np.argmax(cossim)]
                # print(f'add {new_vid}(d{self.temp_degree[new_vid]})', end=';')

                # connect nvid-new_vid, and update status
                self.add_edge.append([nvid, new_vid])
                self.temp_degree[nvid] += 1
                self.temp_degree[new_vid] += 1
                temp_vv[nvid].append(new_vid)
                temp_vv[new_vid].append(nvid)
                self.delete_points[new_vid] = False
                ban_list.append(new_vid)

                if self.temp_degree[new_vid] >= 2:
                    break
                else:
                    nvid = new_vid
            # print('')
        
        if not self.only_extend:
            # delete d1 chain which failed to extend
            vid_d1 = [idx[0] for idx in np.argwhere(self.temp_degree == 1)]
            while(len(vid_d1) > 0):
                vid = vid_d1.pop()
                # print(f'Process {vid}, {self.points[vid]}')
                if self.temp_degree[vid] == 1:
                    chain = self.get_chain(vid, temp_vv)
                    # print(f'length of chain:{len(chain)}, last vertex:{chain[-1]}, deg:{self.temp_degree[chain[-1]]}')
                    self.delete_points[chain[:-1]] = True
                    self.temp_degree[chain[-1]] -= 1
        
        # remove all isolated points
        vid_d0 = np.argwhere(self.temp_degree == 0)
        self.delete_points[vid_d0] = True
        self.__update()
        self.is_clean = True


    @timeout_decorator.timeout(timeout_seconds)
    def construct_endpts_graph(self):
        if not self.is_clean:
            print('Curve is not clean')
            return
        # points with degree larger than 2
        n_pts = self.points.shape[0]
        dl2 = np.arange(n_pts)[self.degree > 2]
        pts_mark = np.zeros(n_pts, dtype=bool)

        dl2_path = []
        temp_dict = {idx:[] for idx in dl2}
        for vid in dl2:
            vid_ns = self.vv[vid]
            for vid_n in vid_ns:
                if self.degree[vid_n] > 2:
                    if vid_n not in temp_dict[vid]:
                        temp_dict[vid].append(vid_n)
                        temp_dict[vid_n].append(vid)
                        dl2_path.append([vid, vid_n])

                    continue

                if not pts_mark[vid_n]:
                    # mark the hole chain except two ends 
                    chain = self.get_d2chain(vid, vid_n, pts_mark)
                    dl2_path.append(chain)

        # topo utils
        dl2_edges = [[path[0], path[-1]] for path in dl2_path]
        dl2_ve = {idx: [] for idx in dl2}
        dl2_n = {idx: [] for idx in dl2}
        for num,edge in enumerate(dl2_edges):
            vid1, vid2 = edge
            dl2_ve[vid1].append(num)
            dl2_ve[vid2].append(num)

            dl2_n[vid1].append(vid2)
            dl2_n[vid2].append(vid1)

        # Handle different cases
        merge_paths = []
        # dl2_degree = {idx: len(vn) for idx,vn in dl2_n.items()}
        delete_paths = []
        # case1: two closed path between two endpoints
        for vid,vid_n in dl2_n.items():
            uni_vid, counts = np.unique(vid_n, return_counts=True)
            uid_vid = uni_vid[counts > 1]
            if len(uid_vid) == 0:
                continue
            eid_paths = dl2_ve[vid]
            # print(f'vid:{vid}, vid_n:{vid_n}, eid_paths:{eid_paths}')
            for next_vid in uid_vid:
                # find all paths between vid and next_vid
                v2next_path_eid = [eid for eid in eid_paths if dl2_path[eid][-1]==next_vid]
                if len(v2next_path_eid) == 0:
                    continue
                closed_eid = self.check_closed_path(dl2_path, v2next_path_eid)
                if closed_eid is not None:
                    eid1, eid2 = closed_eid[0]
                    merge_paths.extend(closed_eid)
                    delete_paths.append(eid2)

        new_paths = [path for eid,path in enumerate(dl2_path) if eid not in delete_paths]
        new_ve = {idx: [] for idx in dl2}
        for eid,path in enumerate(new_paths):
            new_ve[path[0]].append(eid)
            new_ve[path[-1]].append(eid)
        # new_degree = {idx:len(ve) for idx,ve in new_ve.items()}

        # # case 2: dl2 points form a triangle 
        # NOTE: code below is buggy, this case is skipped
        # new_paths = self.update_paths(new_paths, new_ve, new_degree)
        # res = self.handle_triangle(new_paths)

        # endpts must be marked at last since neighbor mark check
        pts_mark[dl2] = True
        # extract closed curves(circles) from remained points
        closed_paths = self.extract_closed_curve(pts_mark)
        new_paths.extend(closed_paths)
        
        return new_paths

    def __update(self):
        # add edges
        if len(self.add_edge) > 0:
            self.edges.extend(self.add_edge)

        # filter deleted points and re-index points and edges
        remain_pid = np.logical_not(self.delete_points)
        old_pidx = np.arange(self.points.shape[0])
        new_pidx = old_pidx[remain_pid]
        pid_dict = {pid:num for num,pid in enumerate(new_pidx)}

        self.points = self.points[remain_pid]
        new_edges = []
        for e in self.edges:
            i,j = e
            if remain_pid[i] and remain_pid[j]:
                new_edges.append([pid_dict[i], pid_dict[j]])

        self.edges = new_edges
        
        self.tree = KDTree(self.points)
        self.vv = self.__calc_neighbor()
        self.degree = np.asarray([len(nv) for nv in self.vv])
        self.delete_points = np.zeros(self.points.shape[0], dtype=bool)
        self.add_edge = []
        self.temp_degree = self.degree.copy()

    def __calc_neighbor(self):
        N_pts = self.points.shape[0]
        neighbors = [[] for i in range(N_pts)]
        for e in self.edges:
            i,j = e
            neighbors[i].append(j)
            neighbors[j].append(i)

        return neighbors


    def load_clean_curve(self, curve_path):
        with open(curve_path, 'rb') as f:
            data = pickle.load(f)
        
        self.points = data['points']
        self.edges = data['edges']

        self.tree = KDTree(self.points)
        self.vv = self.__calc_neighbor()
        self.degree = np.asarray([len(nv) for nv in self.vv])
        self.delete_points = np.zeros(self.points.shape[0], dtype=bool)
        self.add_edge = []
        self.temp_degree = self.degree.copy()
        self.is_clean = True

    def check_closed_path(self, dl2_path, eid):
        # check if paths of eid exists closed(geometry) groups
        paths = [dl2_path[idx] for idx in eid]
        nums = [len(path) for path in paths]
        res = []
        # Now assume only exists one-pair of paths
        for path_pair in combinations(range(len(eid)), 2):
            idx1,idx2 = path_pair
            if abs(nums[idx1] - nums[idx2]) < 4:
                path1, path2 = paths[idx1], paths[idx2]
                dist = self.path_mean_dist(path1, path2)
                # if dist < 2*self.step:
                if dist < self.closed_path_thres:
                    res.append([eid[idx1], eid[idx2]])
        if len(res) == 0:
            return None
        else:
            return res

    def extract_closed_curve(self, pts_mark):
        res = []
        nonmark = [vid for vid in range(self.points.shape[0]) if not pts_mark[vid]]
        while not np.all(pts_mark):
            pid = nonmark[0]
            closed = self.get_d2closed(pid, pts_mark)
            nonmark = [vid for vid in nonmark if not pts_mark[vid]]
            res.append(closed)

        return res

    def __vertex_edge_from_path(self, paths):
        ve = {}
        vvn = {}
        for num,path in enumerate(paths):
            pid, qid = path[0], path[-1]
            if pid in ve:
                ve[pid].append(num)
                vvn[pid].append(qid)
            else:
                ve[pid] = [num]
                vvn[pid] = [qid]
            
            if pid == qid:
                continue

            if qid in ve:
                ve[qid].append(num)
                vvn[qid].append(pid)
            else:
                ve[qid] = [num]
                vvn[qid] = [pid]
        
        degree = {idx: len(es) for idx,es in ve.items()}
        return ve, vvn, degree

    def __remove_repeated_path(self, paths):
        def is_path_same(path1, path2):
            if len(path1) != len(path2):
                return False
            path1 = np.asarray(path1)
            path2 = np.asarray(path2)
            return np.all(path1==path2) or np.all(path1==path2[::-1])

        ve, vvn, degree = self.__vertex_edge_from_path(paths)
        delete_path = []
        for pid, pid_n in vvn.items():
            unique, counts = np.unique(pid_n, return_counts=True)
            if len(unique) < len(pid_n):
                # may exists repeated paths
                qids = unique[counts > 1]
                for qid in qids:
                    pq_paths = np.intersect1d(ve[pid], ve[qid])
                    # print(f'pid:{pid}, qid:{qid}, pq_paths:{pq_paths}')
                    assert len(pq_paths) == 2, 'pq_paths more than 2'
                    if is_path_same(paths[pq_paths[0]], paths[pq_paths[1]]):
                        delete_path.extend(pq_paths[1:])

        new_paths = [paths[i] for i in range(len(paths)) if i not in delete_path]
        return new_paths

    def path_dist(self, path):
        diff = self.points[path[:-1]] - self.points[path[1:]]
        diff = np.linalg.norm(diff, axis=1)
        return np.sum(diff)

    def handle_triangle(self, paths):
        def check_triangle(_vid_n):
            for pair in combinations(_vid_n, 2):
                vid1, vid2 = pair
                if vid2 in vvn[vid1]:
                    return pair
        def path_idx(_pid, _qid):
            # here we can assume only one path in p~q, since previous merging operation
            int_path = np.intersect1d(ve[_pid], ve[_qid])
            if len(int_path) > 1:
                delete_path.extend(int_path[1:])
            return int_path[0]
        def choose_delete(_eids):
            # choose edge with max lengths
            dists = [self.path_dist(paths[eid]) for eid in _eids]
            return _eids[np.argmax(dists)]

        delete_path = []
        ve, vvn, degree = self.__vertex_edge_from_path(paths)
        # remove_paths = self.__remove_repeated_path(paths, ve, vvn, degree)
        # delete_path.extend(remove_paths)
        triangles = []
        vflag = {idx:False for idx in ve.keys()}
        for Aid,Aid_n in vvn.items():
            if vflag[Aid]:
                continue
            if degree[Aid] != 3:
                continue
            
            tri_idx = check_triangle(Aid_n)
            if tri_idx is None:
                continue
            # if triangle(pid, Bid, Cid); Did: another path away from pid 
            Bid, Cid = tri_idx
            eids = [path_idx(Aid, Bid), path_idx(Bid, Cid), path_idx(Cid, Aid)]
            # epaths = [paths[eid] for eid in eids]
            triangles.append([(Aid,Bid,Cid), eids])
            vflag[Aid] = True
            vflag[Bid] = True
            vflag[Cid] = True
        
        if len(triangles) == 0:
            return None

        while (len(triangles) > 0):
            tri, eids = triangles.pop()
            # the delete path(pid, qid)
            eid = choose_delete(eids)
            if len(paths[eid]) < self.delete_triangle_length:
                delete_path.append(eid)

        new_paths = [paths[i] for i in range(len(paths)) if i not in delete_path]
        ve, vvn, degree = self.__vertex_edge_from_path(new_paths)
        return new_paths, ve, degree


    def path_mean_dist(self, path1, path2):
        pts1 = self.points[path1]
        pts2 = self.points[path2]
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

        dist_12, _ = tree2.query(pts1)
        dist_21, _ = tree1.query(pts2)

        return (np.mean(dist_12) + np.mean(dist_21)) / 2

    def update_paths(self, paths, path_ve, deg):
        def check_path(path1, path2):
            if path1[0] == path2[0] and path1[1] == path2[1]:
                return True
            elif path1[0] == path2[-1] and path1[1] == path2[-2]:
                return True
            elif path1[-1] == path2[-1] and path1[-2] == path2[-2]:
                return True
            else:
                return False

        # merge paths(whose endpts becomes d2)
        marks = np.zeros(self.points.shape[0], dtype=bool)
        new_paths = []
        problem_paths = []
        for eid,path in enumerate(paths):
            pid,qid = path[0], path[-1]
            if deg[pid] > 2 and deg[qid] > 2:
                new_paths.append(path)
                # marks[path[1:-1]] = True
                marks[path] = True
            else:
                problem_paths.append(path)
        
        while len(problem_paths) > 0:
            path = problem_paths.pop()
            # print(f'path:{path[0]}->{path[-1]}, num:{len(path)}, all_marks:{np.all(marks[path])}')
            # if np.all(marks[path[1:-1]]):
            if np.all(marks[path]):
                continue

            pid, qid = path[0], path[-1]
            pe = path_ve[pid]
            qe = path_ve[qid]
            # peid = pe[0] if pe[1] == eid else pe[1]
            # qeid = qe[0] if qe[1]==eid else qe[1]
            if deg[pid] == 2 and deg[qid] == 2:
                peid = pe[1] if check_path(paths[pe[0]], path) else pe[0]
                qeid = qe[1] if check_path(paths[qe[0]], path) else qe[0]
                p_path = paths[peid]
                q_path = paths[qeid]
                # check if it is circle
                if pid == q_path[0] or pid == q_path[-1]:
                    # circle case, merge two paths into a circle path
                    if path[-1] != q_path[0]:
                        q_path = q_path[::-1]
                    merge_path = np.concatenate([path[:-1], paths[qeid][::-1]])
                    new_paths.append(merge_path)
                    # marks[merge_path[1:-1]] = True
                    marks[merge_path] = True
                    # print(f'Found circle: {pid}->{qid}')
                else:
                    if p_path[0] == pid:
                        p_path = p_path[::-1]
                    if q_path[0] != qid:
                        q_path = q_path[::-1]
                    
                    merge_path = np.concatenate([p_path[:-1], path, q_path[1:]])
                    if deg[p_path[0]] > 2 and deg[q_path[-1]] > 2:
                        new_paths.append(merge_path)
                        # marks[merge_path[1:-1]] = True
                        marks[merge_path] = True
                    else:
                        problem_paths.append(merge_path)

            if deg[pid] == 2 and deg[qid] > 2:
                peid = pe[1] if check_path(paths[pe[0]], path) else pe[0]
                p_path = paths[peid]
                if p_path[0] == pid:
                    p_path = p_path[::-1]
                merge_path = np.concatenate([p_path[:-1], path])
                if deg[p_path[0]] > 2:
                    new_paths.append(merge_path)
                    # marks[merge_path[1:-1]] = True
                    marks[merge_path] = True
                else:
                    problem_paths.append(merge_path)

            if deg[qid] == 2 and deg[pid] > 2:
                qeid = qe[1] if check_path(paths[qe[0]], path) else qe[0]
                q_path = paths[qeid]
                if q_path[0] != qid:
                    q_path = q_path[::-1]

                merge_path = np.concatenate([path, q_path[1:]])
                if deg[q_path[-1]] > 2:
                    new_paths.append(merge_path)
                    # marks[merge_path[1:-1]] = True
                    marks[merge_path] = True
                else:
                    problem_paths.append(merge_path)
        
        return new_paths

    def get_chain(self, vid, temp_vv):
        # vid: temp degree 1 vertex, temp_vv: temp vertex neighbors
        chain = [vid]
        prev_vid = vid
        vid = temp_vv[vid][0]
        chain.append(vid)
        while (len(temp_vv[chain[-1]]) == 2):
            idx1, idx2 = temp_vv[chain[-1]]
            vid = idx1 if idx2 == prev_vid else idx2
            prev_vid = chain[-1]
            chain.append(vid)

        return chain

    def get_d2chain(self, vid, vid_n, marks):
        # vid dl2 point, vid_n: neighbor of vid
        chain = [vid, vid_n]
        while (self.degree[chain[-1]] == 2):
            idx1, idx2 = self.vv[chain[-1]]
            next_vid = idx1 if idx2 == chain[-2] else idx2
            chain.append(next_vid)

            if self.degree[chain[-1]] < 2:
                raise ValueError('Degree < 2, not clean')
        
        marks[chain[1:-1]] = True
        return chain

    def get_d2closed(self, vid, marks):
        vid_n = self.vv[vid][0]
        closed = [vid, vid_n]
        while closed[-1] != vid:
            idx1, idx2 = self.vv[closed[-1]]
            next_vid = idx1 if idx2 == closed[-2] else idx2
            closed.append(next_vid)

        marks[closed] = True
        return closed

    def batch_cos_similarity(self, v1, v2):
        # v1:(N,3), v2:(N,3)
        v1_normalized = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_normalized = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        return np.einsum('ij,ij->i', v1_normalized, v2_normalized)
        # return (v1 @ v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))

    def path_smoothing(self, paths, niter):
        new_points = self.points.copy()
        for path in paths:
            points_path = new_points[path]
            # do not smooth 2-segment or closed curve
            if len(path) == 2 or path[0] == path[-1]:
                continue
            for i in range(niter):
                points_path[1:-1] = (points_path[0:-2] + points_path[2:]) / 2
            
            new_points[path] = points_path

        return new_points


    def CAD_fitting(self, points, paths, output_path):
        def gen_chain_edge(num_pts):
            edges = np.arange(num_pts)
            edges = np.vstack([edges[:-1], edges[1:]]).T
            return edges.tolist()

        k = 2
        res = {}
        pca = PCA(n_components=2)
        for num, path in enumerate(paths):
            if len(path) == 2:
                res[f'curve_{num}'] = {
                    'points': points[path],
                    'edges': [[0, 1]],
                    'type': 'Line',
                    'closed': False,
                    'err': 0
                }
                continue
            
            is_closed = path[0] == path[-1]
            points_path = points[path]
            if is_closed:
                curve = self.__circle_fitting(points_path, pca)
                if curve['status']:
                    n_curvesamples = curve['points'].shape[0]
                    curve['edges'] = gen_chain_edge(n_curvesamples)
                    res[f'curve_{num}'] = curve
                    continue
            else:
                curve = self.__line_fitting(points_path, num_samples=len(path))
                if curve['status']:
                    n_curvesamples = curve['points'].shape[0]
                    curve['edges'] = gen_chain_edge(n_curvesamples)
                    res[f'curve_{num}'] = curve
                    continue

            # use spline to fit
            num_samples = len(path) + 3
            if is_closed:
                num_samples = max(num_samples, 24)
            curve = self.__BSpline_fitting(points_path, k, num_samples)
            curve['closed'] = is_closed
            n_curvesamples = curve['points'].shape[0]
            curve['edges'] = gen_chain_edge(n_curvesamples)
            res[f'curve_{num}'] = curve

        # endpoints extraction
        endpts_idx = np.asarray([[path[0], path[-1]] for path in paths if path[0] != path[-1]])
        endpts_idx = np.unique(endpts_idx.flatten())
        if len(endpts_idx) > 0:
            endpts = points[endpts_idx]
        else:
            endpts = []
        res['endpoints'] = endpts
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)


    def __line_fitting(self, points, num_samples):
        line_vec = points[-1] - points[0]
        line_vec /= np.linalg.norm(line_vec)

        pts = points[1:-1]
        v0 = points[0]
        # calculate distance of pts to line
        diff = pts - v0
        dist = np.sum((diff)**2, axis=1)
        dist -= (diff @ line_vec)**2
        dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
        if dist.mean() < self.line_thres:
            # return Line samples
            ts = np.linspace(0,1, num_samples)
            return {
                'points': v0 + ts[:,None]*(points[-1]-v0),
                'err': dist.mean(),
                'type': 'Line',
                'closed': False,
                'status': True
            }
        else:
            return {'status': False}

    def __circle_fitting(self, points, pca):
        n_pts = points.shape[0]
        pts2d = pca.fit_transform(points)
        # fit with 2D circle
        A = np.ones((n_pts, 3))
        A[:,:2] = pts2d
        b = np.sum(pts2d**2, axis=1)
        res = np.linalg.lstsq(A, b, rcond=None)[0]
        a,b,c = res
        center = res[0:2] / 2
        radius = np.sqrt(4*c + a**2 + b**2) / 2

        err = np.linalg.norm(pts2d - center, axis=1) - radius
        err = np.abs(err).mean()

        if err < self.circle_thres:
            # sample the circle
            num_samples = max(24, n_pts)
            ts = (2*np.pi)* np.linspace(0,1, num_samples)
            circle = np.vstack([np.cos(ts), np.sin(ts)]).T
            samples2d = center + radius*circle
            samples = pca.inverse_transform(samples2d)
            return {
                'points': samples,
                'err': err,
                'type': 'Circle',
                'closed': True,
                'status': True
            }
        else:
            return {'status': False}

    
    def __BSpline_fitting(self, points, k, num_samples):
        n_pts = points.shape[0]
        num_int_knots = int(n_pts / 2)
        interal_knots = np.linspace(0,1, num_int_knots)[1:-1]
        knots = np.r_[[0.]*(k+1), interal_knots, [1.]*(k+1)]

        xs = np.arange(n_pts, dtype=float)
        xs /= np.max(xs)
        ys = points

        spline = spi.make_lsq_spline(xs, ys, knots, k)
        samples = spline(np.linspace(0,1, num_samples))
        ys_spl = spline(xs)
        err = np.linalg.norm(ys-ys_spl, axis=1).mean()
        samples[0] = ys[0]
        samples[-1] = ys[-1]

        return {
            'points': samples,
            'err': err,
            'type': 'BSpline',
        }

    def export_endpts_graph(self, pidx, pedges, output_path):
        pidx_dict = {idx: num for num,idx in enumerate(pidx)}
        pd = pidx_dict
        out_edges = [[ pd[e[0]], pd[e[1]] ] for e in pedges]
        points = self.points[pidx]
        res = {'points': points, 'edges': out_edges}
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)

    def export_curve(self, output_path):
        res = {'points': self.points, 'edges': self.edges}
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)
        
    
    def export_paths(self, paths, output_path):
        endpts_idx = np.asarray([[path[0],path[-1]] for path in paths])
        endpts_idx = np.unique(endpts_idx.flatten())
        edge_idx = np.concatenate([path[1:-1] for path in paths]).astype(int)
        pts_idx = np.concatenate([endpts_idx, edge_idx])
        reidx_dict = {idx:num for num,idx in enumerate(pts_idx)}

        edges = []
        for path in paths:
            path_reidx = [reidx_dict[idx] for idx in path]
            path_edges = np.vstack([path_reidx[:-1], path_reidx[1:]]).T
            edges.extend(path_edges.tolist())
        
        res = {'points': self.points[pts_idx], 'edges': edges}
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)


    def export_smoothed_paths(self, points, paths, output_path):
        endpts_idx = np.asarray([[path[0], path[-1]] for path in paths])
        endpts_idx = np.unique(endpts_idx.flatten())
        edge_idx = np.concatenate([path[1:-1] for path in paths]).astype(int)
        pts_idx = np.concatenate([endpts_idx, edge_idx])
        reidx_dict = {idx: num for num, idx in enumerate(pts_idx)}

        edges = []
        for path in paths:
            path_reidx = [reidx_dict[idx] for idx in path]
            path_edges = np.vstack([path_reidx[:-1], path_reidx[1:]]).T
            edges.extend(path_edges.tolist())

        res = {'points': points[pts_idx], 'edges': edges}
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)

        endpts_path = output_path.replace('.pkl', '.ply')
        endpts = self.points[endpts_idx]
        mesh = trimesh.Trimesh(endpts, process=False)
        mesh.export(endpts_path)



def generate_CAD_curves():
    root_path = op.dirname(op.abspath(__file__))
    root_path = op.dirname(op.dirname(root_path))
    exp_name = 'demo'
    result_path = op.join(root_path, exp_name)
    print('Result path: ', result_path)
    names = np.loadtxt(op.join(result_path, 'data_list.txt'), dtype=str)

    param = {
        'query_ball_radius': 4,
        'extend_max_count': 6,
        'delete_max_length': 5,
        'closed_path_dist': 2,
        'only_extend': False
    }

    failed_list = []
    t0 = time()
    for count, name in enumerate(names):
        print(f'Processing {count}:{name}')

        pred_curve_path = os.path.join(result_path, name, 'pred_nerve_pwl_curve.pkl')
        output_cad_file = os.path.join(result_path, name, 'cad_curves.pkl')
        output_cad_pwl_file = os.path.join(result_path, name, 'cad_pwl_curve.pkl')
        os.makedirs(os.path.dirname(output_cad_file), exist_ok=True)

        try:
            curve = PWLCurve(pred_curve_path)
            curve.set_parameters(param)
            curve.curve_cleaning()

            new_paths = curve.construct_endpts_graph()
            new_points = curve.path_smoothing(new_paths, niter=3)
            curve.CAD_fitting(new_points, new_paths, output_cad_file)
            convert_cad_to_pwl(output_cad_file, output_cad_pwl_file)
        except Exception as e:
            failed_list.append(name)
            print(f'Problem: {name} err {e}')

    time_cost = time()-t0
    print(f'Done, failed number:{len(failed_list)}, time cost:{time_cost}')

if __name__ == '__main__':
    generate_CAD_curves()