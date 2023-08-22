import bpy,bmesh
import os, glob, pickle
import os.path as op
from time import time

t0 = time()

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

mat = bpy.data.materials.new('black')
mat.diffuse_color = (0., 0., 0., 1.)
cam = bpy.data.objects['Camera']
cam.data.lens = 100
bpy.context.scene.render.resolution_percentage = 50
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

count = 0

root_path = op.dirname(op.abspath(__file__))
root_path = op.dirname(op.dirname(root_path))
dataset_path = op.join(root_path, 'demo')
for data_path in glob.glob(op.join(dataset_path, '**/*.pkl'), recursive=True):
    file = op.basename(data_path)
    if 'pwl_curve' not in file:
        continue

    name, ext = op.splitext(file)
    dirname = op.dirname(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    pts = data['points']
    edges = data['edges']

    new_mesh = bpy.data.meshes.new(f'mesh_{name}')
    new_mesh.from_pydata(pts, edges, [])
    new_mesh.update()

    new_object = bpy.data.objects.new(f'object_{name}', new_mesh)
    scene = bpy.context.scene
    scene.collection.objects.link(new_object)
    bpy.context.view_layer.objects.active = new_object

    new_object.select_set(True)
    bpy.ops.object.convert(target='CURVE')

    new_object.data.bevel_depth = 0.003

    new_object.data.materials.append(mat)
    new_object.active_material_index = 0

    output_path = op.join(dirname, f'{name}.png')
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print('#-----------------------------------#')
    print(f'{name} rendered')
    print('#-----------------------------------#')

    bpy.data.objects.remove(new_object, do_unlink=True)
    bpy.data.meshes.remove(new_mesh, do_unlink=True)

print(f'Done, time cost: {time()-t0}')
