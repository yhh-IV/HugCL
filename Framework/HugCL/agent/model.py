import torch
import torch.nn as nn
import numpy as np
from cpprb import ReplayBuffer
from agent.dqn import EnsembleDQN
from torch.autograd import Variable


class EnsembleAgent:

    def __init__(self, action_size, fre_ego_dim=4, fre_obs_dim=24, tr_dim=4, fre_d2g_dim=2,
                 device='None', learning_rate=5e-3, gamma=0.9, tau=0.01, buffer_size=int(5e5)):
        
        self.fre_ego_dim = fre_ego_dim
        self.fre_obs_dim = fre_obs_dim
        self.tr_dim = tr_dim
        self.fre_d2g_dim = fre_d2g_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.qvals = None

        self.replay_buffer_expert = ReplayBuffer(5e5,
                                        {"fre_ego": {"shape": 4, "dtype": np.float32},
                                         "fre_obs": {"shape": 24, "dtype": np.float32},
                                         "tr": {"shape": 4, "dtype": np.float32},
                                         "fre_d2g": {"shape": 2, "dtype": np.float32},
                                         "act": {},
                                         "rew": {},
                                         "next_fre_ego": {"shape": 4, "dtype": np.float32},
                                         "next_fre_obs": {"shape": 24, "dtype": np.float32},
                                         "next_tr" : {"shape": 4, "dtype": np.float32},
                                         "next_fre_d2g": {"shape": 2, "dtype": np.float32},
                                         "done": {}})

        self.ratio = 0
        
        self.device = device if device != 'None' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = EnsembleDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim, action_size).to(self.device)
        self.target_model = EnsembleDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim, action_size).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.continual_optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()
        hard_update(self.target_model, self.model)
        
    def choose_action_single(self, fre_ego, fre_obs, tr, fre_d2g):
        fre_ego = torch.FloatTensor(fre_ego).float().unsqueeze(0).to(self.device)
        fre_obs = torch.FloatTensor(fre_obs).float().unsqueeze(0).to(self.device)
        tr = torch.FloatTensor(tr).float().unsqueeze(0).to(self.device)
        fre_d2g = torch.FloatTensor(fre_d2g).float().unsqueeze(0).to(self.device)
        self.qvals = self.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        action = np.argmax(self.qvals.cpu().detach().numpy())
        return action
    
    def choose_action_batch(self, fre_ego, fre_obs, tr, fre_d2g):
        fre_ego = torch.FloatTensor(fre_ego).float().unsqueeze(0).to(self.device)
        fre_obs = torch.FloatTensor(fre_obs).float().unsqueeze(0).to(self.device)
        tr = torch.FloatTensor(tr).float().unsqueeze(0).to(self.device)
        fre_d2g = torch.FloatTensor(fre_d2g).float().unsqueeze(0).to(self.device)
        self.qvals = self.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        action = np.argmax(self.qvals.cpu().detach().numpy())
        return action
    
    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).mean()
    
    def learn_offline(self, batch_data):
        data = batch_data
        fre_ego, fre_obs, tr, fre_d2g = data['fre_ego'], data['fre_obs'], data['tr'], data['fre_d2g']
        next_fre_ego, next_fre_obs, next_tr, next_fre_d2g = data['next_fre_ego'], data['next_fre_obs'], data['next_tr'], data['next_fre_d2g'] 
        action, action_exp, reward = data['act'], data['act'], data['rew']
        done = data['done']

        fre_ego = torch.FloatTensor(fre_ego).to(self.device)
        fre_obs = torch.FloatTensor(fre_obs).to(self.device)
        tr = torch.FloatTensor(tr).to(self.device)
        fre_d2g = torch.FloatTensor(fre_d2g).to(self.device)
        next_fre_ego = torch.FloatTensor(next_fre_ego).to(self.device)
        next_fre_obs = torch.FloatTensor(next_fre_obs).to(self.device)
        next_tr = torch.FloatTensor(next_tr).to(self.device)
        next_fre_d2g = torch.FloatTensor(next_fre_d2g).to(self.device)
        
        action = torch.LongTensor(action).to(self.device)
        action_exp = torch.LongTensor(action_exp).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # calculate bellman error
        curr_as = self.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        curr_Q = curr_as.gather(1, action_exp).squeeze(1)
        
        with torch.no_grad():
            next_Q = torch.max(self.target_model.forward(next_fre_ego, next_fre_obs, next_tr, next_fre_d2g), 1)[0]
            expected_Q = reward.squeeze(1) + self.gamma * next_Q * (1 - done.squeeze(1))

        loss = self.MSE_loss(curr_Q, expected_Q.detach())
        c_loss = self.cql_loss(curr_as, action_exp) # calculate cql loss
        q_loss = 0.5 * loss + c_loss # calculate total loss
        
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
            
        soft_update(self.target_model, self.model, self.tau)
    
        return q_loss.item()
    
    def calc_af_loss(self, model):
        af_loss = []
        af_losses = []
        af_lambda = [5e-8, 5e-8, 5e-8, 5e-8, 5e-8]
        learners = [self.model.learner_1, self.model.learner_2, self.model.learner_3, 
                    self.model.learner_4, self.model.learner_5]
        for idx, learner in enumerate(learners):
            for n, p in learner.named_parameters():
                if p.requires_grad:
                    af_loss.append((p.pow(2)).sum())
            af_losses.append(0.5 * af_lambda[idx] * sum(af_loss)) 
        return sum(af_losses)
    
    def calc_sp_loss(self, model, old_model, fisher):
        sp_loss = []
        sp_lambda = 1e6
        for (n, p), (_, old_p) in zip(self.model.named_parameters(), old_model.named_parameters()):
            if p.requires_grad:
                sp_loss.append((fisher[n] * (p - old_p).pow(2)).sum())
        return 0.5 * sp_lambda * sum(sp_loss)
    
    def conlearn_offline(self, batch_data, old_model, fisher):
        data = batch_data
        fre_ego, fre_obs, tr, fre_d2g = data['fre_ego'], data['fre_obs'], data['tr'], data['fre_d2g']
        next_fre_ego, next_fre_obs, next_tr, next_fre_d2g = data['next_fre_ego'], data['next_fre_obs'], data['next_tr'], data['next_fre_d2g'] 
        action, action_exp, reward = data['act'], data['act'], data['rew']
        done = data['done']

        fre_ego = torch.FloatTensor(fre_ego).to(self.device)
        fre_obs = torch.FloatTensor(fre_obs).to(self.device)
        tr = torch.FloatTensor(tr).to(self.device)
        fre_d2g = torch.FloatTensor(fre_d2g).to(self.device)
        next_fre_ego = torch.FloatTensor(next_fre_ego).to(self.device)
        next_fre_obs = torch.FloatTensor(next_fre_obs).to(self.device)
        next_tr = torch.FloatTensor(next_tr).to(self.device)
        next_fre_d2g = torch.FloatTensor(next_fre_d2g).to(self.device)
        
        action = torch.LongTensor(action).to(self.device)
        action_exp = torch.LongTensor(action_exp).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # calculate bellman error
        curr_as = self.model.forward(fre_ego, fre_obs, tr, fre_d2g)
        curr_Q = curr_as.gather(1, action_exp).squeeze(1)
        
        with torch.no_grad():
            # human-guided Q advantage term
            HQ_as = self.model.forward(fre_ego, fre_obs, tr, fre_d2g)
            HQ_adv = torch.exp(HQ_as.gather(1, action_exp)).squeeze(1) / torch.exp(HQ_as).sum(dim=1)
            
            next_Q = torch.max(self.model.forward(next_fre_ego, next_fre_obs, next_tr, next_fre_d2g), 1)[0]
            expected_Q = reward.squeeze(1) + self.gamma * next_Q * (1 - done.squeeze(1)) + HQ_adv
            
        loss = self.MSE_loss(curr_Q, expected_Q.detach())

        
        c_loss = self.cql_loss(curr_as, action_exp) # cql loss
        sp_loss = self.calc_sp_loss(self.model, old_model, fisher)  # memory stability loss
        af_loss = self.calc_af_loss(self.model) # active forgetting loss
        q_loss = 0.5 * loss + c_loss + sp_loss + af_loss # total loss
        
        self.continual_optimizer.zero_grad()
        q_loss.backward()
        self.continual_optimizer.step()

        return q_loss.item()    
    
    def store_transition(self, ego, obs, tr, d2g, a, r, ego_, obs_, tr_, d2g_, d=0):
        self.replay_buffer_expert.add(fre_ego=ego,
                fre_obs=obs,
                tr=tr,
                fre_d2g=d2g,
                act=a,
                rew=r,
                next_fre_ego=ego_,
                next_fre_obs=obs_,
                next_tr=tr_,
                next_fre_d2g=d2g_,
                done=d)
      
    def load_model(self, output):
        if output is None: return
        self.model.load_state_dict(torch.load('{}.pkl'.format(output)))
        hard_update(self.target_model, self.model)

    def save_model(self, output, timeend=0):
        torch.save(self.model.state_dict(), '{}/{}.pkl'.format(output, timeend))
    
    def save_transition(self, output, timeend=0):
        self.replay_buffer_expert.save_transitions(file='{}/{}.npz'.format(output, timeend))
                
    def load_transition(self, output):
        if output is None: return
        self.replay_buffer_expert.load_transitions('{}.npz'.format(output))
    
    def append_transition_expert(self):
        samples = self.replay_buffer_expert.get_all_transitions()
        self.replay_buffer_expert.add(fre_ego=samples["fre_ego"],
                fre_obs=samples["fre_obs"],
                tr=samples["tr"],
                fre_d2g=samples["fre_d2g"],
                act=samples["act"],
                rew=samples["rew"],
                next_fre_ego=samples["next_fre_ego"],
                next_fre_obs=samples["next_fre_obs"],
                next_tr=samples["next_tr"],
                next_fre_d2g=samples["next_fre_d2g"],
                done=samples["done"])  
    
    def clear_transition(self):
        self.replay_buffer_expert.clear()
    
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)