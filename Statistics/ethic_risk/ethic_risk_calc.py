import pickle
import torch
import numpy as np
from utils.collision_probability import get_collision_probability_fast, get_inv_mahalanobis_dist
from utils.harm_estimation import get_harm
from utils.frenet_functions import FrenetTrajectory
from configs.vehicleparams import VehicleParameters 
from configs.load_json import load_harm_parameter_json, load_risk_json
from predictor.model import WaleNet

def calc_risk(traj, pred, ego_params, obs_params, params, cov):  
    """
    ethical risk = collision harm * collision probability.
    """
    
    modes = params['modes']
    coeffs = params['harm']
    
    # calculate collsion probability
    if modes["fast_prob_mahalanobis"]:
        coll_prob_dict = get_inv_mahalanobis_dist(traj=traj, pred=pred, cov=cov)
    else:
        coll_prob_dict = get_collision_probability_fast(traj=traj, pred=pred, ego_params=ego_params, obs_params=obs_params, cov=cov)

    # calculate road user harm
    ego_harm_traj, obs_harm_traj = get_harm(traj, pred, ego_params, obs_params, modes, coeffs)
    
    # Calculate risk out of harm and collision probability
    ego_risk_traj = {}
    obs_risk_traj = {}
    ego_risk_max = {}
    obs_risk_max = {}

    # ethic_risk = harm * prob 
    ego_risk_traj = [ego_harm_traj[t] * coll_prob_dict[t] for t in range(len(ego_harm_traj))]
    obs_risk_traj = [obs_harm_traj[t] * coll_prob_dict[t] for t in range(len(obs_harm_traj))]
    
    # Take max as representative for the whole trajectory
    ego_risk_max = max(ego_risk_traj)
    obs_risk_max = max(obs_risk_traj)
    
    return ego_risk_max, obs_risk_max

def main():
    traj_path = './traj_data.pkl'
    model = WaleNet(input_dim=4, hidden_dim=128, num_layers=4, output_dim=5, output_timesteps=20)
    
    ### load trained model ###
    model.load_state_dict(torch.load('./predictor/predictor.pth'))
    
    with open(traj_path, 'rb') as file:
        traj_data = pickle.load(file)
    
    all_data = []
    global ethic_risk_record
    ethic_risk_record = []
    
    for step in range(len(traj_data)):
        step_data = {}
        step_data["ego_s"] = traj_data[step][0]
        step_data["ego_d"] = traj_data[step][1]
        step_data["ego_yaw"] = traj_data[step][2]
        step_data["ego_v"] = traj_data[step][3]
        step_data["obs_s"] = traj_data[step][4]
        step_data["obs_d"] = traj_data[step][5]
        step_data["obs_yaw"] = traj_data[step][6]
        step_data["obs_v"] = traj_data[step][7]
        all_data.append(step_data)
    
    print("All data have been loaded !!!")
    
    ego_traj = FrenetTrajectory()
    obs_traj = FrenetTrajectory()
    
    # load parameters
    ego_params = VehicleParameters("tesla")
    obs_params = VehicleParameters("human")  # ford_escort, vw_vanagon, human
    params_harm = load_harm_parameter_json()
    params_weights = None
    params_mode = load_risk_json()
    params_dict = {'weights': params_weights, 'modes': params_mode, 'harm': params_harm}
    
    for step in range(len(all_data)):
        step_data = all_data[step]
        ego_traj.s = np.array(step_data['ego_s'])
        ego_traj.d = np.array(step_data['ego_d'])
        ego_traj.yaw = np.array(step_data['ego_yaw'])
        ego_traj.v = np.array(step_data['ego_v'])
        
        inputs = np.array((step_data['obs_s'][0], step_data['obs_d'][0], step_data['obs_yaw'][0], step_data['obs_v'][0]))
        inputs = torch.tensor(np.tile(inputs, (20, 1)))
        inputs = inputs.unsqueeze(0).float()
        with torch.no_grad():
            outputs = model(inputs).squeeze()
        
        obs_traj.s = np.array(outputs[:, 0]) + step_data['obs_s'][0]
        obs_traj.d = np.array(outputs[:, 1]) + step_data['obs_d'][0]
        obs_traj.yaw = np.array(step_data['obs_yaw'])
        obs_traj.v = np.array(step_data['obs_v'])
        
        sigma_s = np.array(torch.exp(torch.clamp(outputs[0, 2], min=-1, max=1))) 
        sigma_d = np.array(torch.exp(torch.clamp(outputs[0, 3], min=-1, max=1)))
        sigma_sd = np.array(torch.clamp(outputs[0, 4], min=-0.5, max=0.5))
        obs_cov = [[sigma_s, sigma_sd], [sigma_sd, sigma_d]]
        
        ego_risk_max, obs_risk_max = calc_risk(
            traj=ego_traj,
            pred=obs_traj,
            ego_params=ego_params,
            obs_params=obs_params,
            params=params_dict,
            cov=obs_cov)
        
        # calculate the overall risk            
        ethic_risk_record.append(ego_risk_max + obs_risk_max)
        
        
if __name__ == "__main__":
    main() 