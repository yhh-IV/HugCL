import torch
import torch.nn as nn

class HyDuelingDQN(nn.Module):
    def __init__(self, fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim):
        super(HyDuelingDQN, self).__init__()
        
        veh_dim = fre_ego_dim + fre_obs_dim
        env_dim = tr_dim + fre_d2g_dim
        
        self.veh_proj = nn.Sequential(
            nn.Linear(veh_dim, 128),
            nn.ReLU()
        ) 
        
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, 32),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128+32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128+32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
                
    def forward(self, fre_ego, fre_obs, tr, fre_d2g):
        veh = torch.cat((fre_ego, fre_obs), dim=1)
        veh_feature = self.veh_proj(veh)
        env = torch.cat((tr, fre_d2g), dim=1)
        env_feature = self.env_proj(env)
        syn_feature = torch.cat((veh_feature, env_feature), dim=1)
        values_feature = self.value_stream(syn_feature)
        advantages_feature = self.advantage_stream(syn_feature)

        return values_feature, advantages_feature


class EnsembleDQN(nn.Module):
 
    def __init__(self, fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim, output_dim):
        super(EnsembleDQN, self).__init__()
        
        self.learner_1 = HyDuelingDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim)
        self.learner_2 = HyDuelingDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim)
        self.learner_3 = HyDuelingDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim)
        self.learner_4 = HyDuelingDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim)
        self.learner_5 = HyDuelingDQN(fre_ego_dim, fre_obs_dim, tr_dim, fre_d2g_dim)
        
        self.values_proj = nn.Linear(128, output_dim)
        self.advantages_proj = nn.Linear(128, 1)

    def forward(self, fre_ego, fre_obs, tr, fre_d2g):

        vf1, adf1 = self.learner_1.forward(fre_ego, fre_obs, tr, fre_d2g)
        vf2, adf2 = self.learner_2.forward(fre_ego, fre_obs, tr, fre_d2g)
        vf3, adf3 = self.learner_3.forward(fre_ego, fre_obs, tr, fre_d2g)
        vf4, adf4 = self.learner_4.forward(fre_ego, fre_obs, tr, fre_d2g)
        vf5, adf5 = self.learner_5.forward(fre_ego, fre_obs, tr, fre_d2g)
   
        vfs = torch.stack([vf1, vf2, vf3, vf4, vf5]) # leaner_num * batch size * features
        adfs = torch.stack([adf1, adf2, adf3, adf4, adf5])
        values_features = torch.sum(vfs, dim=0).squeeze(0)
        advantage_features = torch.sum(adfs, dim=0).squeeze(0)
        
        values = self.values_proj(values_features) 
        advantages = self.advantages_proj(advantage_features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals