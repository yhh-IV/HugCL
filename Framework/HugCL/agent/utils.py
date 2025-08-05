import sys
import torch
from torch.utils.data import DataLoader

from data_loader import DictDataset

def calc_init_fisher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from algo.ens_d3qn import EnsembleAgent as DRL    
    action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    DRL = DRL(action_size=len(action_list), device=device, buffer_size=int(5e5))
    # prepare transition data and trained model
    DRL.replay_buffer.clear()
    try: 
        DRL.replay_buffer.load_transitions("../offline_RL/CrossDataset.npz")
        print("Old dataset has been loaded !!!")
    except:
        print("Error! Check old dataset existing !!!")
    dataset = DictDataset(DRL.replay_buffer.get_all_transitions())
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    try:
        DRL.model.load_state_dict(torch.load('{}.pkl'.format('../offline_RL/algo/models/ens_dqn/base_model')))
        print("Old model has been loaded !!!")
    except:
        print("Error! Check old model existing !!!")
    
    fisher = {}
    for n, p in DRL.model.named_parameters():
        fisher[n] = 0 * p.data
    
    for batch_data in dataloader:
        data = batch_data
        fre_ego, fre_obs, tr, fre_d2g = data['fre_ego'], data['fre_obs'], data['tr'], data['fre_d2g']
        next_fre_ego, next_fre_obs, next_tr, next_fre_d2g = data['next_fre_ego'], data['next_fre_obs'], data['next_tr'], data['next_fre_d2g'] 
        action, action_exp, reward, interv = data['act'], data['acte'], data['rew'], data['intervene']
        done = data['done']
    
        fre_ego = torch.FloatTensor(fre_ego).to(device)
        fre_obs = torch.FloatTensor(fre_obs).to(device)
        tr = torch.FloatTensor(tr).to(device)
        fre_d2g = torch.FloatTensor(fre_d2g).to(device)
        next_fre_ego = torch.FloatTensor(next_fre_ego).to(device)
        next_fre_obs = torch.FloatTensor(next_fre_obs).to(device)
        next_tr = torch.FloatTensor(next_tr).to(device)
        next_fre_d2g = torch.FloatTensor(next_fre_d2g).to(device)
        
        action = torch.LongTensor(action).to(device)
        action_exp = torch.LongTensor(action_exp).to(device)
        reward = torch.FloatTensor(reward).to(device)
        interv = torch.FloatTensor(interv).to(device)
        done = torch.FloatTensor(done).to(device)
    
        # calculate bellman error
        curr_as = DRL.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        curr_Q = curr_as.gather(1, action_exp).squeeze(1)
    
        with torch.no_grad():
            next_Q = torch.max(DRL.target_model.forward(next_fre_ego, next_fre_obs, next_tr, next_fre_d2g), 1)[0]
            expected_Q = reward.squeeze(1) + DRL.gamma * next_Q * (1 - done.squeeze(1))

        loss = DRL.MSE_loss(curr_Q, expected_Q.detach())
        c_loss = DRL.cql_loss(curr_as, action_exp)
        q_loss = 0.5 * loss + c_loss
    
        DRL.optimizer.zero_grad()
        q_loss.backward()
        for n, p in DRL.model.named_parameters():
            if p.grad is not None:
                fisher[n] += done.size(0) * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n, _ in DRL.model.named_parameters():
            fisher[n] = fisher[n] / len(dataset)
          
    return fisher

def update_fisher(old_fisher, t, phi):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from algo.ens_d3qn import EnsembleAgent as DRL    
    action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    DRL = DRL(action_size=len(action_list), device=device, buffer_size=int(5e5))
    # prepare transition data and trained model
    model_name =  'cl_s' + str(t) + '_model'
    data_name = 'CL_S' + str(t) + '_Dataset'
    try:
        DRL.model.load_state_dict(torch.load('{}.pkl'.format('../offline_RL/algo/models/ens_dqn/' + model_name)))
        print("Model %s has been loaded !!!" % (str(t)))
    except:
        print("Error! Check model %s existing !!!" % (str(t)))
    try: 
        DRL.replay_buffer.clear()
        DRL.replay_buffer.load_transitions('{}.npz'.format("../offline_RL/" + data_name))
        print("Dataset %s has been loaded !!!" % (str(t)))
    except:
        print("Error! Check dataset %s existing !!!" % (str(t)))
    dataset = DictDataset(DRL.replay_buffer.get_all_transitions())
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    fisher = {}
    for n, p in DRL.model.named_parameters():
        fisher[n] = 0 * p.data
    
    for batch_data in dataloader:
        data = batch_data
        fre_ego, fre_obs, tr, fre_d2g = data['fre_ego'], data['fre_obs'], data['tr'], data['fre_d2g']
        next_fre_ego, next_fre_obs, next_tr, next_fre_d2g = data['next_fre_ego'], data['next_fre_obs'], data['next_tr'], data['next_fre_d2g'] 
        action, action_exp, reward, interv = data['act'], data['acte'], data['rew'], data['intervene']
        done = data['done']
    
        fre_ego = torch.FloatTensor(fre_ego).to(device)
        fre_obs = torch.FloatTensor(fre_obs).to(device)
        tr = torch.FloatTensor(tr).to(device)
        fre_d2g = torch.FloatTensor(fre_d2g).to(device)
        next_fre_ego = torch.FloatTensor(next_fre_ego).to(device)
        next_fre_obs = torch.FloatTensor(next_fre_obs).to(device)
        next_tr = torch.FloatTensor(next_tr).to(device)
        next_fre_d2g = torch.FloatTensor(next_fre_d2g).to(device)
        
        action = torch.LongTensor(action).to(device)
        action_exp = torch.LongTensor(action_exp).to(device)
        reward = torch.FloatTensor(reward).to(device)
        interv = torch.FloatTensor(interv).to(device)
        done = torch.FloatTensor(done).to(device)
    
        # calculate bellman error
        curr_as = DRL.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        curr_Q = curr_as.gather(1, action_exp).squeeze(1)
    
        with torch.no_grad():
            next_Q = torch.max(DRL.target_model.forward(next_fre_ego, next_fre_obs, next_tr, next_fre_d2g), 1)[0]
            expected_Q = reward.squeeze(1) + DRL.gamma * next_Q * (1 - done.squeeze(1))

        loss = DRL.MSE_loss(curr_Q, expected_Q.detach())
        c_loss = DRL.cql_loss(curr_as, action_exp)
        q_loss = 0.5 * loss + c_loss
    
        DRL.optimizer.zero_grad()
        q_loss.backward()
        for n, p in DRL.model.named_parameters():
            if p.grad is not None:
                fisher[n] += done.size(0) * p.grad.data.pow(2)  # batch_size=256
    # Mean
    with torch.no_grad():
        for n, _ in DRL.model.named_parameters():
            fisher[n] = fisher[n] / len(dataset)
            
    # Online update
    with torch.no_grad():
        for n, _ in DRL.model.named_parameters():
            fisher[n] = phi * old_fisher[n] + (1 - phi) * fisher[n]
            
    return fisher