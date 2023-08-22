from OCC.Extend.TopologyUtils import TopologyExplorer, discretize_edge
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.GCPnts import GCPnts_UniformAbscissa, GCPnts_UniformDeflection
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Dir, gp_Pln
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepLProp import BRepLProp_CurveTool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

# bounding box
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

# intersection
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCCUtils.edge import Edge

import os, pickle, glob
import numpy as np
from time import time
import yaml


def get_BB(shape, tol=1e-6, use_mesh=True):
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center = np.asarray([(xmin+xmax)/2., (ymin+ymax)/2., (zmin+zmax)/2.])
    scale = max([xmax-xmin, ymax-ymin, zmax-zmin]) / 2
    return center, scale

def normalize(shape, offset, factor):
    T = gp_Trsf()
    x, y, z = offset
    T.SetTranslation(gp_Vec(x, y, z))
    loc = TopLoc_Location(T)
    shape.Move(loc)

    T = gp_Trsf()
    origin = gp_Pnt(0,0,0)
    T.SetScale(origin, factor)
    loc = TopLoc_Location(T)
    shape.Move(loc)
    return shape


def edge_samples(reso, file_path, output_path):
    def get_pos(pnt):
        return [pnt.X(), pnt.Y(), pnt.Z()]

    # name, ext = os.path.splitext(os.path.basename(file_path))
    output_file = os.path.join(output_path, f'edge_all_reso{reso}.pkl')
    if os.path.exists(output_file):
        return
    shapes = read_step_file(file_path, verbosity=False)
    shapes = TopoDS_Shape(shapes)
    center, scale = get_BB(shapes)
    print(f'center: {center.tolist()}, scale: {scale}')
    offset = -center
    factor = 0.9 / scale
    shapes = normalize(shapes, offset, factor)

    step = 1.0/ reso
    topExp = TopologyExplorer(shapes, ignore_orientation=True)
    print('Num of edges: ', topExp.number_of_edges())
    samples = {}
    for k, edge in enumerate(topExp.edges()):
        edge = Edge(edge)
        res = {}
        res['is_closed'] = edge.is_closed()
        res['type'] = edge.type
        
        if edge.type == 'line':
            edge_length = factor*edge.length()
        else:
            # edge_samples = edge.edge_uniform_samples(step)
            edge_length = edge.length()

        num = int(edge_length / step) + 1
        edge_samples = edge.divide_by_number_of_points(num)
        if edge_samples is None:
            continue
        res['parameters'] = [t for t,p in edge_samples]
        res['samples'] = [get_pos(p) for t,p in edge_samples]

        samples[f'edge{k}'] = res


    verts = []
    for vert in topExp.vertices():
        pos = get_pos(BRep_Tool.Pnt(vert))
        verts.append(pos)

    samples['vertices'] = verts
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)

    print('Done, saved to', output_path)


def sharp_edge_samples(reso, step_path, feat_path, output_path):
    def get_pos(pnt):
        return [pnt.X(), pnt.Y(), pnt.Z()]

    # name, ext = os.path.splitext(os.path.basename(step_path))
    output_file = os.path.join(output_path, f'step_edge_reso{reso}.pkl')
    if os.path.exists(output_file):
        return

    shapes = read_step_file(step_path, verbosity=False)
    shapes = TopoDS_Shape(shapes)
    center, scale = get_BB(shapes)
    # print(f'center: {center.tolist()}, scale: {scale}')
    offset = -center
    factor = 0.9 / scale
    shapes = normalize(shapes, offset, factor)

    with open(feat_path) as f:
        feat = yaml.load(f, Loader=yaml.CLoader)

    step = 1.0/ reso
    topExp = TopologyExplorer(shapes, ignore_orientation=True)
    # print('Num of edges: ', topExp.number_of_edges())
    samples = {}
    n_othercurve = 0
    for k, edge in enumerate(topExp.edges()):
        edge = Edge(edge)

        if edge.type == 'othercurve':
            n_othercurve += 1
            continue

        is_sharp = feat['curves'][k-n_othercurve]['sharp']
        if not is_sharp:
            continue

        res = {}
        res['is_closed'] = edge.is_closed()
        res['type'] = edge.type
        
        if edge.type == 'line':
            edge_length = factor*edge.length()
        else:
            edge_length = edge.length()

        num = int(edge_length / step) + 1
        edge_samples = edge.divide_by_number_of_points(num)

        if edge_samples is None:
            continue
        res['parameters'] = [t for t,p in edge_samples]
        res['samples'] = [get_pos(p) for t,p in edge_samples]

        samples[f'edge{k}'] = res


    verts = []
    for vert in topExp.vertices():
        pos = get_pos(BRep_Tool.Pnt(vert))
        verts.append(pos)

    samples['vertices'] = verts
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)

    return samples


def samples2curve(data, output_path):
    points = []
    edges = []

    head = 0
    for key in data:
        if 'edge' not in key:
            continue

        edge_data = data[key]
        pts = np.asarray(edge_data['samples'])
        points.append(pts)

        start_idx = np.arange(head, head+pts.shape[0])
        next_idx = np.roll(start_idx, -1)
        if not edge_data['is_closed']:
            next_idx = next_idx[:-1]
            start_idx = start_idx[:-1]

        edge_idx = np.vstack((start_idx, next_idx))
        edges.append(edge_idx.T)

        head += pts.shape[0]

    points = np.concatenate(points, axis=0)
    edges = np.concatenate(edges, axis=0)

    res = {
        'points': points, 
        'edges': edges
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    reso = 64
    step_ds_path = 'Path/to/your/ABC/v0000'
    feat_ds_path = 'Path/to/your/ABC/abc_0000_feat_v00'
    output_path = 'Path/to/your/ABC/NerVE64Dataset'
    os.makedirs(output_path, exist_ok=True)

    data_list = 'Path/to/your/ABC/NerVE64Dataset/all.txt'
    data_list = np.loadtxt(data_list, dtype=str)

    t0 = time()
    for count, fname in enumerate(data_list):
        # fname = '%08d' % i
        if count % 100 == 0:
            print(f'processing {count}: {fname}')

        step_path = os.path.join(step_ds_path, fname)
        name = os.listdir(step_path)[0]
        step_path = os.path.join(step_path, name)

        feat_path = os.path.join(feat_ds_path, fname)
        if len(os.listdir(feat_path)) > 0:
            name = os.listdir(feat_path)[0]
        else:
            print('No feature file, continue')
            continue
        feat_path = os.path.join(feat_path, name)

        out_file_path = os.path.join(output_path, fname)
        os.makedirs(out_file_path, exist_ok=True)
        # edge_samples(reso, step_path, out_file_path)
        samples = sharp_edge_samples(reso, step_path, feat_path, out_file_path)
        
        out_step_curve_path = os.path.join(out_file_path, 'step_curve_no_offset.pkl')
        samples2curve(samples, out_step_curve_path)


    print(f'Done, total time cost: {time()-t0}')