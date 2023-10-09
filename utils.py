
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

skeleton_links_female = [
    "F_RUpperArm", "F_LUpperArm", "F_LThigh", "F_RThigh"
]


skeleton_links_male = [
    "M_RUpperArm", "M_LUpperArm", "M_Hips", "M_LThigh", "M_RThigh"
]

IDS = ['Hoyer_Base_L', 'M_Head', 'Hoyer_Base_R', 'M_Torso', 'C_Head', 
 'Hoyer_Top', 'M_RCalf', 'F_RCalf', 'M_RThigh', 'Bed_Foot', 'PPS_L', 'F_LThigh', 
 'Bedpan', 'Wheelchair', 'Commode', 'M_Hips', 'M_LUpperArm', 'M_RUpperArm', 'M_RLowerArm', 
 'F_RLowerArm', 'F_Torso', 'F_LCalf', 'F_RThigh', 'F_LUpperArm', 'F_Head', 'Bed_Head', 'M_LThigh', 
 'PPS_R', 'F_Hips', 'M_LCalf', 'Brush', 'M_LLowerArm', 'F_RUpperArm', 'F_LLowerArm']

def load_markers(filename):
    starting_directory = f'joint_positions/{filename}.txt'
    marker_positions = {}
    
    with open(starting_directory, 'r') as f:
        line = f.readline()
        found_start = False
        rigid_set = ""
        while line != '':
            if not found_start:
                if "RigidBody Name" not in line:
                    line = f.readline()
                    continue
                else:
                    rigid_set = line.split('RigidBody Name : ')[1].replace('\n', '')
                    found_start = True
                    marker_positions[rigid_set] = []
            else:
                if "Data Description" in line:
                    found_start = False
                else:
                    if "Position" in line:
                        position = line.split("Position: ")[1].replace('\n', '')
                        x, y, z = position.split(", ")
                        marker_positions[rigid_set].append(list(map(float, [x, y, z])))
            line = f.readline()

    return marker_positions

def visualize_keypoints(joint_type, show = False):
    marker_positions = load_markers(18)
    markers = np.array(marker_positions[joint_type])
    ### because the markers are all initialized when the manikin is on the bed in a fixed position, we can simply 
    ### use the min/max in the Z-direction to obtain the front/end of the body markers
    ### meanwhile for the torso, the two top most -Z will form the left/right shoulder, and the midpoint can be 
    ### extrapolated into the midsection
    
    if "Torso" in joint_type:
        forward_most = np.argmin(markers[:, 0])
        back_most = np.argmax(markers[:, 0])
    else:
        forward_most = np.argmax(markers[:, 2])
        back_most = np.argmin(markers[:, 2])
    colours = {
        forward_most: [255, 0, 0, 255],
        back_most: [0, 255, 0, 255]
    }
    
    if show:
        scene = trimesh.Scene()
        for i in range(len(markers)):
            colour = colours[i] if i in colours else [0, 0, 0, 255]
            marker = markers[i]
            cloud = trimesh.points.PointCloud(np.array([marker]), colors=np.tile(np.array(colour), (1, 1)))
            scene.add_geometry(cloud)
        return scene
    else:
        return markers[forward_most], markers[back_most]

def make_transformation_matrix(pose):
    # assumes lst is in the form of [x, y, z, ox, oy, oz, ow]
    pos_x, pos_y, pos_z, or_x, or_y, or_z, or_w = pose
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R.from_quat([or_x, or_y, or_z, or_w]).as_matrix()
    transformation_matrix[:3, 3] = [pos_x, pos_y, pos_z]
    return transformation_matrix
    
def get_point(pose, joint_name):
    # returns the <front point> and <back point>
    pos_x, pos_y, pos_z, or_x, or_y, or_z, or_w = pose
    transformation_world_to_obj = make_transformation_matrix(pose)

    transformation_centroid_to_offshoot_front = np.eye(4)
    transformation_centroid_to_offshoot_back = np.eye(4)

    offshoot_forward, offshoot_back = visualize_keypoints(joint_name)
    transformation_centroid_to_offshoot_front[:3, 3] = offshoot_forward
    transformation_centroid_to_offshoot_back[:3, 3] = offshoot_back

    transform_world_to_front = transformation_world_to_obj @ transformation_centroid_to_offshoot_front
    transform_world_to_back = transformation_world_to_obj @ transformation_centroid_to_offshoot_back

    front_point = (transform_world_to_front @ np.array([0, 0, 0, 1]))[:3].T
    back_point = (transform_world_to_back @ np.array([0, 0, 0, 1]))[:3].T
    
    return front_point, back_point

def skeleton_reconstruct(skeleton_positions, gender):
    # input: dictionary {ID: [general_position]}
    joints = skeleton_links_female if gender == 0 else skeleton_links_male
    lines = []
    centroids = []
    print(joints)
    for joint_2 in joints:
        try:
            
            pos_x, pos_y, pos_z, or_x, or_y, or_z, or_w = skeleton_positions[joint_2]
            centroids.append([pos_x, pos_y, pos_z])
            front_point, back_point = get_point(skeleton_positions[joint_2], joint_2)
            
#             if list(np.array(front_point).nonzero()[0]) == [] or list(np.array(back_point).nonzero()[0]) == []:
#                 continue

            lines.append([front_point, back_point, joint_2, joint_2])
        except Exception as e:
            print(e)
            continue
    formatter = "F" if gender == 0 else "M"
    if f"{formatter}_LUpperArm" in skeleton_positions and f"{formatter}_RUpperArm" in skeleton_positions:
        max_L, min_L = get_point(skeleton_positions[f"{formatter}_LUpperArm"], f"{formatter}_LUpperArm")
        max_R, min_R = get_point(skeleton_positions[f"{formatter}_RUpperArm"], f"{formatter}_RUpperArm")
        
        lines.append([min_L, min_R, f"{formatter}_LShoulder", f"{formatter}_RShoulder"])
        print("Added shoulder-shoulder links")
        
        # for the lower arm we will snap it to the end point of the upper arms directly
        if f"{formatter}_LLowerArm" in skeleton_positions and f"{formatter}_RLowerArm" in skeleton_positions:
            max_LL, min_LL = get_point(skeleton_positions[f"{formatter}_LLowerArm"], f"{formatter}_LLowerArm")
            max_RL, min_RL = get_point(skeleton_positions[f"{formatter}_RLowerArm"], f"{formatter}_RLowerArm")
            
            lines.append([max_L, max_LL, f"{formatter}_LElbow", f"{formatter}_LWrist"])
            lines.append([max_R, max_RL, f"{formatter}_RElbow", f"{formatter}_RWrist"])
        
        if f"{formatter}_Hips" in skeleton_positions:
            mid_arms = (min_L + min_R) / 2
            x, y, z, _, _, _, _ = skeleton_positions[f"{formatter}_Hips"]
            # for greater accuracy we will use the point at the crotch so we change the Z value to the max
            max_hips, _ = get_point(skeleton_positions[f"{formatter}_Hips"], f"{formatter}_Hips")
            z = max_hips[2]
            
            lines.append([mid_arms, [x, y, z], f"{formatter}_Torso", f"{formatter}_Hips"])
            print("Added torso-hip links")
            
            if f"{formatter}_LThigh" in skeleton_positions and f"{formatter}_RThigh" in skeleton_positions:
                _, min_L = get_point(skeleton_positions[f"{formatter}_LThigh"], f"{formatter}_LThigh")
                _, min_R = get_point(skeleton_positions[f"{formatter}_RThigh"], f"{formatter}_RThigh")
                
                
                lines.append([min_L, [x, y, z], f"{formatter}_LThigh", f"{formatter}_Hips"])
                lines.append([min_R, [x, y, z], f"{formatter}_RThigh", f"{formatter}_Hips"])
                print("Added hip-thigh links")
            
    if f"{formatter}_LThigh" in skeleton_positions and f"{formatter}_RThigh" in skeleton_positions:
        max_L, min_L = get_point(skeleton_positions[f"{formatter}_LThigh"], f"{formatter}_LThigh")
        max_R, min_R = get_point(skeleton_positions[f"{formatter}_RThigh"], f"{formatter}_RThigh")
        
        # snap the calf line from the knee to the ankle
        if f"{formatter}_LCalf" in skeleton_positions and f"{formatter}_RCalf" in skeleton_positions:
            max_LL, min_LL = get_point(skeleton_positions[f"{formatter}_LCalf"], f"{formatter}_LCalf")
            max_RL, min_RL = get_point(skeleton_positions[f"{formatter}_RCalf"], f"{formatter}_RCalf")
            
            lines.append([max_L, max_LL, f"{formatter}_LKnee", f"{formatter}_LAnkle"])
            lines.append([max_R, max_RL, f"{formatter}_RKnee", f"{formatter}_RAnkle"])
    return lines, centroids