import os,pickle
import numpy as np
from scipy.spatial import KDTree

def load_cad_curve(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    edge_points = []

    for name,val in data.items():
        if name == 'endpoints':
            continue
        points = val['points']
        edge_points.append(points[1:-1])

    edge_points = np.concatenate(edge_points, axis=0)

    endpts = data['endpoints']
    points = np.concatenate([edge_points, endpts], axis=0)

    return points

def convert_cad_to_pwl(data_path, output_path):
    def get_edge(_v0, _v1, _vidx):
        e1 = np.r_[_v0, _vidx]
        e2 = np.r_[_vidx, _v1]
        e12 = np.vstack([e1, e2]).T.astype(int)
        return e12.tolist()

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    edges = []
    edge_points = []
    endpts = data['endpoints']
    if len(endpts) > 0:
        tree_endpts = KDTree(endpts)
    else:
        tree_endpts = None

    current_idx = len(endpts)
    for name, val in data.items():
        if name == 'endpoints':
            continue
        
        points = val['points']
        is_closed = val['closed']
        if is_closed:
            edge_points.append(points)
            pts_idx = list(range(current_idx, current_idx+points.shape[0]))
            next_idx = np.roll(pts_idx, -1)
            tmp_edges = np.vstack([pts_idx, next_idx]).T.astype(int)
            edges.extend(tmp_edges)
            current_idx += points.shape[0]
            continue
        
        if tree_endpts is None:
            raise ValueError('Not endpts in open curves')

        num_inpts = points.shape[0] - 2
        _,v0 = tree_endpts.query(points[0])
        _,v1 = tree_endpts.query(points[-1])
        if num_inpts == 0:
            edges.append([v0,v1])
            continue

        edge_points.append(points[1:-1])
        pts_idx = list(range(current_idx, current_idx+num_inpts))
        tmp_edges = get_edge(v0, v1, pts_idx)
        edges.extend(tmp_edges)
        current_idx += num_inpts

    edge_points = np.concatenate(edge_points, axis=0)
    if len(endpts) > 0:
        points = np.concatenate([endpts, edge_points], axis=0)
    else:
        points = edge_points
    res = {
        'points': points,
        'edges': edges
    }
    with open(output_path, 'wb') as f:
        pickle.dump(res, f)


def load_pwl_curve(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    pts, edges = data['points'], data['edges']
    midpts = np.mean(pts[np.asarray(edges)], axis=1)

    samples = np.concatenate([pts, midpts], axis=0)
    return samples


def load_step_curve(data_path, offset=None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    pts, edges = data['points'], data['edges']
    midpts = np.mean(pts[np.asarray(edges)], axis=1)

    samples = np.concatenate([pts, midpts], axis=0)
    if offset is not None:
        samples += offset

    return samples

def calc_loss(pred, gt, max_HD=False):
    """
    pred: (N, 3) samples of predicted curve
    gt: (M,3) samples of gt curve
    """
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)

    dist_pred2gt, _ = tree_gt.query(pred)
    dist_gt2pred, _ = tree_pred.query(gt)

    chamfer_dist = np.mean(dist_pred2gt**2) + np.mean(dist_gt2pred**2)
    if max_HD:
        bhaussdorf_dist = max(dist_pred2gt.max(), dist_gt2pred.max())
    else:
        bhaussdorf_dist = (dist_pred2gt.max() + dist_gt2pred.max()) / 2
    return {
        'CD': chamfer_dist,
        'BHD': bhaussdorf_dist
    }


if __name__ == "__main__":
    pred_curve_path = '/path/to/your/pred_nerve_pwl_curve.pkl'
    # pred_cad_curve_path = '/path/to/your/cad_pwl_curve.pkl'
    gt_nerve_curve_path = '/path/to/your/nerve_reso64_curve.pkl'
    gt_step_path = '/path/to/your/step_curve_no_offset.pkl'

    offset = pickle.load(open(gt_nerve_curve_path, 'rb'))['stable_offset']
    pred_pwl_samples = load_pwl_curve(pred_curve_path)
    # pred_pwl_samples = load_pwl_curve(pred_cad_curve_path)
    gt_step_samples = load_step_curve(gt_step_path, offset)

    err = calc_loss(pred_pwl_samples, gt_step_samples)
    