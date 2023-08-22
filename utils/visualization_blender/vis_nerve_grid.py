import bpy, bmesh
import numpy as np
import os, pickle
import os.path as op

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)


root_path = '/home/uhzoaix/Project/nerve'
data_path = op.join(root_path, 'demo', '00000274', 'pred_nerve.pkl')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

k = data['grid_size']
step = 2./ k

cube_index = data['cube_idx']
cube_faces = data['cube_faces']

blender_cube_faces = np.asarray([0, 3, 4])
# Materials
if 'basic' in bpy.data.materials and 'blue' in bpy.data.materials:
    mat0 = bpy.data.materials['basic']
    mat_blue = bpy.data.materials['blue']
else:
    mat0 = bpy.data.materials.new('basic')
    mat0.diffuse_color = (1., 0., 0., 0.8)
    mat_blue = bpy.data.materials.new('blue')
    mat_blue.diffuse_color = (0.130295, 0.242092, 1, 0.8)

for idx, face in zip(cube_index, cube_faces):
    loc = step*idx + (step/2. - 1.)
    bpy.ops.mesh.primitive_cube_add(size=step, location=tuple(loc))

    new_cube = bpy.context.active_object
    new_cube.name = 'Cube_(%d,%d,%d)' % (idx[0], idx[1], idx[2])

    new_cube.data.materials.append(mat0)
    new_cube.data.materials.append(mat_blue)
    new_cube.active_material_index = 0

    # i,j,k = idx
    for polygon_idx in blender_cube_faces[face]:
        new_cube.data.polygons[polygon_idx].material_index = 1
