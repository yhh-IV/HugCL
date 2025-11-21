import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../HugCL/navigation/')
from cubic_spline_planner import LocalPlanner

with open('human_traj.pkl', 'rb') as f:
    traj_info = pickle.load(f)

# traj_info = np.load('./human_traj.npy')

def clac_dis(traj, target_point):
    dis = np.linalg.norm(traj - target_point, axis=1)
    return dis

def calc_ADE(traj, target_traj):
    dis = np.mean(np.linalg.norm(traj - target_traj[:len(traj)], axis=1))
    return dis

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

planner = LocalPlanner()
# traj_info = traj_info[:, 240:]
# data loader
ego_s = traj_info["s"]
ego_d = traj_info["d"]
ego_x = traj_info["x"]
ego_y = traj_info["y"]
ego_yaw = traj_info["yaw"] 
ego_v = traj_info["v"] 
timestamp = traj_info["ts"]

# data sample 0.5s
diff_time = np.cumsum(np.diff(timestamp))
idx = np.where(np.abs(diff_time / 0.5 - np.round(diff_time / 0.5)) < 0.01)
spl_s = ego_s[idx]
spl_d = ego_d[idx]

start_s = spl_s[0]
start_d = spl_d[0]
start_yaw = ego_yaw
start_v = ego_v
start_dec = 5

full_dec = []
full_s = []
full_d = []
full_v = []
full_s.append(start_s)
full_d.append(start_d)
full_v.append(start_v)

# set the threshold
thre_dis = 0.1
lane_width = 3.5
temp_loc = None
junction_check = None
horizon = 1
opt_traj_num = 5
opt_traj = [[] for _ in range(opt_traj_num)]

# initialize the optimal trajs
for traj_id in range(opt_traj_num):
    opt_traj[traj_id].append([start_s, start_d, start_yaw, start_v, start_dec])

### optimal neighbor search ###
for i in range(len(spl_s)-1):
    temp_traj_list = []
    traj_ADEs = []
    fig = plt.figure(figsize=(15 / 2.54, 2.5 / 2.54))  # 15cm, 2.5cm
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axhline(y=1.75, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=5.25, color='black', linestyle='--', linewidth=1)
    for traj_id in range(opt_traj_num):
        next_s = []
        next_d = []
        next_yaw = []
        next_v = []
        
        for dec in range(9):
            curr_s = opt_traj[traj_id][-1][0]
            curr_d = opt_traj[traj_id][-1][1]
            curr_yaw = opt_traj[traj_id][-1][2]
            curr_v = opt_traj[traj_id][-1][3]
            # determine the lane id
            if curr_d < (0.5 * lane_width):
                curr_id = 0
            elif curr_d > (1.5 * lane_width):
                curr_id = 2
            else:
                curr_id = 1
        
            if dec in [0, 3, 6]:
                curr_v = round(np.clip(curr_v + 2, 0, 30), 2)
            elif dec in [2, 5, 8]:
                curr_v = round(np.clip(curr_v - 2, 0, 30), 2)
            
            target_dis = curr_v / 3.6 * 0.5
            target_s = curr_s + round(np.clip(curr_v*0.8, 5, 15), 2)
            
            if dec in [0, 1, 2]:  # left turn 
                target_id = np.clip(curr_id+1, 0, 2)
            elif dec in [3, 4, 5]: # keep
                target_id = curr_id
            else: # right turn
                target_id = np.clip(curr_id-1, 0, 2)
            
            target_d = target_id * lane_width
            
            curr_loc = np.array((curr_s, curr_d))
            curr_pos = curr_yaw
            targ_loc = np.array((target_s, target_d))
            targ_pos = 0
            
            traj = planner.path_generation(curr_loc, curr_pos, temp_loc, targ_loc, targ_pos, junction_check)
            dis = np.linalg.norm(traj - curr_loc, axis=1)
            
            traj_yaw = np.arctan2(np.diff(traj[:, 1]), np.diff(traj[:, 0]))
            traj_yaw = wrap_angle(np.insert(traj_yaw, 0, traj_yaw[-1]))
            
            idx = np.argmin(np.abs(dis - target_dis))
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.8, zorder=10)    
            
            next_s.append(traj[idx][0])
            next_d.append(traj[idx][1])
            next_yaw.append(traj_yaw[idx])
            next_v.append(curr_v)
           
        ### beam search ###
        candidate_dis = clac_dis(np.column_stack((next_s, next_d)), np.array((spl_s[i+1], spl_d[i+1])))
        closest_idxs = np.where((candidate_dis - min(candidate_dis)) < thre_dis)[0]
    
        for closest_idx in closest_idxs:
            # Append the decision for the current closest index
            full_dec.append(closest_idx)
            # Append the next state to the temporary trajectory list
            temp_traj_list.append(opt_traj[traj_id] + [[next_s[closest_idx], next_d[closest_idx], 
                                                        next_yaw[closest_idx], next_v[closest_idx],
                                                        closest_idx]])

    # filter the optimal trajectory
    for temp_traj_id in range(len(temp_traj_list)):
        temp_traj = np.array(temp_traj_list[temp_traj_id])[1:, :2]
        traj_ADE = calc_ADE(temp_traj, np.array((spl_s, spl_d)).T)
        traj_ADEs.append(traj_ADE)
    
    # Select the two trajs with the smallest ADE
    min_idxs = sorted(range(len(traj_ADEs)), key=lambda x: traj_ADEs[x])[:opt_traj_num]
    
    for traj_id, min_idx in enumerate(min_idxs):
        opt_traj[traj_id] = temp_traj_list[min_idx]
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('none')
    ax.scatter(spl_s, spl_d, color=np.array((206, 208, 209))/255)
    ax.plot(spl_s, spl_d, color=np.array((206, 208, 209))/255, linewidth=2)
    ax.scatter(np.array(opt_traj[traj_id])[:, 0], np.array(opt_traj[traj_id])[:, 1], color=np.array((255, 192, 0))/255)
    ax.plot(np.array(opt_traj[traj_id])[:, 0], np.array(opt_traj[traj_id])[:, 1], color=np.array((255, 192, 0))/255, linewidth=1.5)
    
    ax.scatter(spl_s[i+1], spl_d[i+1], color=np.array((144, 208, 77))/255) # target
    ax.scatter(next_s[closest_idx], next_d[closest_idx], color=np.array((197, 119, 124))/255)
    
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels():
        label.set_color('white')
    plt.xlabel('', fontname='Arial')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.axis('equal')
    plt.ylim(-1.75, 8.75)
    plt.xlim(spl_s[i+1] - 20, spl_s[i+1] + 25)

    plt.show()
