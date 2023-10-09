
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import time
import pickle
from utils import *


### LOAD DATA
with open('18-0-7.pkl', 'rb') as f:
    data = pickle.load(f)

index = 0
skeleton_lines, centroids = skeleton_reconstruct(data[index][-1], 0)


# GOAL: trimesh objects to the pyrender for quick rendering
T = len(data)#50#*60*10
fps = 15
# save_dir = "rendered_images"


scene = pyrender.Scene()
# pyrender_line=pyrender.Primitive(mode=1, positions=[[0,0,0],[0.2,0.2,0.2]], color_0=[[10,255,255,255],[255,10,255,255]])
# line_mesh = pyrender.Mesh(primitives=[pyrender_line])
# line_node = scene.add(line_mesh)


pyrender_line_nodes = []
for line in skeleton_lines:
    x, y, joint_1, joint_2 = line
    pyrender_line=pyrender.Primitive(mode=1, positions=[x,y], color_0=[[255,0,0,255],[255,0,0,255]])
    line_mesh = pyrender.Mesh(primitives=[pyrender_line])
    line_node = scene.add(line_mesh)
    pyrender_line_nodes.append(line_node)

##### DEFINE PYRENDER MATTERS
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = 1#np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   1.6],
    [1.0,  0.0, 0.0, 1.0],
    [0.0,  s,   s,   2.0],
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
    global pyrender_line_nodes, scene, skeleton_lines

    skeleton_lines, centroids = skeleton_reconstruct(data[i][-1], 0)
    if len(skeleton_lines) == 0:
        color, depth = r.render(scene)
        matplot_img.set_data(color)
        return [matplot_img]

    # update line iteratively

    for line_node in pyrender_line_nodes:
        scene.remove_node(line_node)
        
    pyrender_line_nodes = []
    for line in skeleton_lines:
        x, y, joint_1, joint_2 = line
        pyrender_line=pyrender.Primitive(mode=1, positions=[x,y], color_0=[[255,0,0,255],[255,0,0,255]])
        line_mesh = pyrender.Mesh(primitives=[pyrender_line])
        line_node = scene.add(line_mesh)
        pyrender_line_nodes.append(line_node)

    # scene.remove_node(line_node)
    # positions = np.random.rand(2,3)*0.1
    # pyrender_line=pyrender.Primitive(mode=1, positions=positions, color_0=[[10,255,255,255],[255,10,255,255]])
    # line_mesh = pyrender.Mesh(primitives=[pyrender_line])
    # line_node = scene.add(line_mesh)

    # line_pose = scene.get_pose(line_node)
    # line_pose = R.random().as_matrix()
    # scene.set_pose(line_node, pose=line_pose)

    color, depth = r.render(scene)
    matplot_img.set_data(color)
    return [matplot_img]


Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)  # try -1 as well


start_time = time.time()

ani = animation.FuncAnimation(fig, update, T, interval=1, blit=True)
ani.save('animation_drawing.mp4', writer=writer)

print(f'It took {time.time()-start_time} to generate animation with {T} frames.')



