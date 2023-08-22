import bpy,bmesh
import os, glob, pickle
import os.path as op

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

root_path = '/home/uhzoaix/Project/nerve'
data_path = op.join(root_path, 'demo', '00000274', 'pred_nerve_pwl_curve.pkl')
fname = 'pwl_curve'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

pts = data['points']
edges = data['edges']

new_mesh = bpy.data.meshes.new(f'mesh_{fname}')
new_mesh.from_pydata(pts, edges, [])
new_mesh.update()

new_object = bpy.data.objects.new(f'object_{fname}', new_mesh)
scene = bpy.context.scene
scene.collection.objects.link(new_object)
