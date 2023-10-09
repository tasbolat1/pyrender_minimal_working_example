import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import time

# GOAL: trimesh objects to the pyrender for quick rendering
T = 10
save_dir = "rendered_images"

# create trimesh primitive
basic_trimesh = trimesh.primitives.Box(extents=[0.1,0.1,0.1], transform=np.eye(4))
basic_trimesh.visual.face_colors = [155,155,255,255]

### Comment trimesh scene
# scene = trimesh.Scene()
# scene.add_geometry(basic_trimesh)
# scene.show()


#### CONVERT TRIMESH MESHES TO PYRENDER
# add smooth=False if trimesh face have visual colors
mesh = pyrender.Mesh.from_trimesh(basic_trimesh, smooth=False)
scene = pyrender.Scene()
mesh_node = scene.add(mesh)

#### PRIMITIVES FOR LINES CAN BE CREATED IN PYRENDER
pyrender_line=pyrender.Primitive(mode=1, positions=[[0,0,0],[0.2,0.2,0.2]], color_0=[[10,255,255,255],[255,10,255,255]])
line_mesh = pyrender.Mesh(primitives=[pyrender_line])
line_node = scene.add(line_mesh)

##### DEFINE PYRENDER MATTERS
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)

#### START SAVING IMAGES
fig, ax = plt.subplots()
matplot_img = ax.imshow(color)

def update(i):
    global line_node, scene, mesh_node
    # update pose iteratively
    pose = scene.get_pose(mesh_node)
    pose[:3,:3] = np.dot(pose[:3,:3], R.random().as_matrix())
    scene.set_pose(mesh_node, pose=pose)

    # update line iteratively
    scene.remove_node(line_node)
    positions = np.random.rand(2,3)*0.1
    pyrender_line=pyrender.Primitive(mode=1, positions=positions, color_0=[[10,255,255,255],[255,10,255,255]])
    line_mesh = pyrender.Mesh(primitives=[pyrender_line])
    line_node = scene.add(line_mesh)

    # line_pose = scene.get_pose(line_node)
    # line_pose = R.random().as_matrix()
    # scene.set_pose(line_node, pose=line_pose)

    color, depth = r.render(scene)
    matplot_img.set_data(color)
    return [matplot_img]

start_time = time.time()

ani = animation.FuncAnimation(fig, update, T, interval=1, blit=False)
ani.save('animation_drawing.gif', writer='imagemagick', fps=5)

print(f'It took {time.time()-start_time} to generate animation with {T} frames.')


# What about converting the Trimesh scene fully

