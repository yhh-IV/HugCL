import numpy as np
import pygame
import signal
import argparse
import pickle
import sys
import torch

def inference():
    set_seed(0)
    from env import TestScenarios
    env = TestScenarios(control_interval=args.action_execution_frequency,
                        frame=args.simulator_render_frequency, port=args.simulator_port)
        
    action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    with open('./scenario_runner/scenario.pkl', 'rb') as file:
        scenario = pickle.load(file)
        
    from agent.model import EnsembleAgent as DRL
    DRL = DRL(action_size=len(action_list), device=args.device, buffer_size=int(5e5))
    
    try:
        DRL.model.load_state_dict(torch.load('{}.pkl'.format('./agent/policy')))
    except:
        print('Error! Check model file existing!')
    
    for i in range(args.maximum_episode):
        step = 0
        done = False
    
        spawn_loc = scenario[i]['spawn_loc']
        goal = scenario[i]['goal']
        
        fre_ego, fre_obs, tr, fre_d2g = env.reset(spawn_loc, goal)
        fre_ego_ = fre_ego.copy()
        fre_obs_ = fre_obs.copy()
        tr_ = tr.copy()
        fre_d2g_ = fre_d2g.copy()
        
        while not done:
            ## action execution ##
            if step % 5 == 0:
                action = DRL.choose_action_single(fre_ego, fre_obs, tr, fre_d2g)
             
            ## environment update ##
            reward = 0
            for _ in range(env.control_interval):
                fre_ego_, fre_obs_, tr_, fre_d2g_, human_action, rr, done, scope = env.step(action_list[action])
                reward += rr
                if done:
                    break

            ## data store ##
            if human_action is not None:
                DRL.store_transition(fre_ego, fre_obs, tr, fre_d2g, human_action, reward, fre_ego_, fre_obs_, tr_, fre_d2g_, done)
            
            fre_ego = fre_ego_.copy()
            fre_obs = fre_obs_.copy()
            tr = tr_.copy()
            fre_d2g = fre_d2g_.copy()
            step += 1
            signal.signal(signal.SIGINT, signal_handler)
            
            if args.data_collection and (done == 1):
                DRL.replay_buffer_expert.save_transitions(file='data_examples.npz')
                
    pygame.display.quit()
    pygame.quit()
    
def signal_handler(sig, frame):
    print('Procedure terminated!')
    pygame.display.quit()
    pygame.quit()
    sys.exit(0)
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--maximum_episode', type=float, help='Maximum training episode number (default:50)', default=1)
    parser.add_argument('--action_execution_frequency', type=int, help='Action execution frequency, step (default: 1)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
    parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 12)', default=60)
    parser.add_argument('--data_collection', action='store_true', help='human guidance collection (default: False)', default=False)
    args = parser.parse_args()

    # Run
    inference()