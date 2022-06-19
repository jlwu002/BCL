from __future__ import division
import time
import torch
import torch.nn.functional as F
import random
import logging
import numpy as np
from datetime import datetime
from copy import deepcopy
import torch.nn as nn

from utils import setup_logger
from plotter import plot
from evaluate import eval_greedy_wc, eval_action_pert
from utils import EpsilonScheduler

import os

# BCL loss function (Generate Adversarial Perturbations)
def _compute_robust_loss(curr_model, target_model, data, epsilon, kappa, gamma, device, args, epsilon_end=None):
    state, action, reward, next_state, done, pert_delta = data
    
    q_values      = curr_model(state)
    q_values_pert = curr_model(state + pert_delta)
    next_q_values = curr_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    
    onehot_labels = torch.zeros(q_values.shape).to(device)
    onehot_labels[range(state.shape[0]), action] = 1
    
    upper_diff = q_values_pert - q_values*(1-onehot_labels) - expected_q_value.detach().unsqueeze(1)*onehot_labels
    wc_diff = torch.abs(upper_diff)

    #sum over output layer, mean only in batch dimension
    worst_case_loss = torch.sum(torch.min(wc_diff.pow(2), wc_diff), dim=1).mean()
    
    standard_loss = standard_loss.mean()
    
    loss = (kappa*(standard_loss)+(1-kappa)*(worst_case_loss))
    
    return loss, standard_loss, worst_case_loss

def _compute_loss(curr_model, target_model, data, gamma, device):
    state, action, reward, next_state, done = data
    
    value, advs = curr_model(state)
    q_values = value + advs

    next_value, next_advs = curr_model(next_state)
    next_q_values = next_value + next_advs

    target_next_value, target_next_advs = target_model(next_state)
    target_next_q_values = target_next_value + target_next_advs

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    #Huber loss
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    standard_loss = standard_loss.mean()
    
    return standard_loss, standard_loss, standard_loss

def calculate_grad(curr_model, data, Q_list_learned, epsilon, device):
    #data is a numpy array
    mm = nn.Softmax(dim = 1)
    state = torch.from_numpy(data).to(device)
    #state.requires_grad = True
    delta = torch.zeros_like(state).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad = True
    Q_list_policy = curr_model(state+delta)
    obj_func = torch.sum(mm(Q_list_policy)*Q_list_learned)
    curr_model.zero_grad()
    obj_func.backward()
    grad = delta.grad.detach()
    delta.data = torch.clamp(delta + 0.375 * torch.sign(grad), -epsilon, epsilon)
    delta = delta.detach()
    gradient = -delta.cpu().numpy().copy()
    # returns a numpy array
    return gradient
    
def apply_pgd(curr_model, Q_list_learned, data, device, epsilon,
               steps = 1,
               relative_step_size = 1):
       
        if epsilon == 0:
            return torch.zeros(data.size()).numpy()

        #clone to avoid modify the actual data (state)
        state = data.detach().clone().cpu().numpy().copy()
        state_orig = state.copy()
        min_ = np.maximum(0, state - epsilon)
        max_ = np.minimum(1, state + epsilon)

        step = 1

        gradient = calculate_grad(curr_model, state, Q_list_learned, epsilon,device)
            
        state = state + gradient

        state = np.clip(state, min_, max_)

        pert_delta = state - state_orig
        
        return pert_delta
    
def train(current_model, target_model, env, args):
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    log = {}
    setup_logger('{}_log'.format(args.env), r'{}{}_{}_log'.format(
        args.log_dir, args.env, start_time))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))
    
    
    #linearly decrease epsilon from 1 to epsilon end over epsilon decay steps
    epsilon_start = 1.0
    def action_epsilon_by_frame(frame_idx):
        if frame_idx <= args.exp_epsilon_decay or not args.decay_zero:
            return (args.exp_epsilon_end + max(0, 1-frame_idx/args.exp_epsilon_decay)*(epsilon_start-args.exp_epsilon_end))
        else:
            return max(0, (args.total_frames-frame_idx) / (args.total_frames-args.exp_epsilon_decay))*(args.exp_epsilon_end)
    
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    
    replay_buffer = ReplayBuffer(args.buffer_size, device)
    start = time.time()
    
    losses = []
    standard_losses = []
    worst_case_losses = []
    all_rewards = []
    worst_case_rewards = []
    episode_reward = 0
    attack_epsilon = 0
    
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    #best_avg = float("-inf")
    
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(current_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr, amsgrad=args.amsgrad, eps=args.adam_eps)

    if args.smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", args.replay_initial, args.attack_epsilon_schedule, 0, 
                                                args.attack_epsilon_end, args.attack_epsilon_schedule)
    
    start_step = 1
    if args.resume_training:
        saved_state = torch.load(
            args.resume_training,
            map_location=lambda storage, loc: storage)
        current_model.load_state_dict(saved_state['model_state_dict'])
        target_model.load_state_dict(current_model.state_dict())
        
        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        # move start state back to allow for regenerating a replay buffer without skipping training
        start_step = saved_state['step'] - args.replay_initial

    for frame_idx in range(start_step, args.total_frames + 1):
        action_epsilon = action_epsilon_by_frame(frame_idx)
        action = current_model.act(state, action_epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        #scale rewards between -1 and 1
        reward = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
        done =  torch.FloatTensor([done]).to(device)

        Q_list_learned = target_model(state).detach()
        pert_delta = apply_pgd(current_model, Q_list_learned, state, device, epsilon = attack_epsilon)
        pert_delta = torch.from_numpy(pert_delta).to(device)

        replay_buffer.push(state, action, reward, next_state, done, pert_delta)

        state = next_state
        
        
        if frame_idx%50000==0:
            test_reward = eval_action_pert(current_model, deepcopy(env), args, epsilon=0)
            log['{}_log'.format(args.env)].info("Steps: {}, Test reward: {}, Time taken: {:.2f}s".format(frame_idx, test_reward, time.time()-start))

        if done:
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            all_rewards.append(episode_reward)
            episode_reward = 0
            #plot(frame_idx, all_rewards, losses, standard_losses, worst_case_losses, args, start_time)
            
            
            state_to_save = {'model_state_dict':current_model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'step':frame_idx}
            
            if args.resume_training:
                torch.save(state_to_save, args.resume_training[:-3]+"_last.pt")
            else:
                torch.save(state_to_save, '{}{}_{}_last.pt'.format(
                    args.save_model_dir, args.env, start_time))
        
        if len(replay_buffer) > args.replay_initial and frame_idx%(args.batch_size/args.updates_per_frame)==0:
            
            init_coeff = (frame_idx - args.replay_initial +1)/min(args.attack_epsilon_schedule, args.total_frames)
            #clip between 0 and 1
            lin_coeff = max(min(1, init_coeff), 0)
            
            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff*args.attack_epsilon_end
            
            attack_epsilon = attack_epsilon + args.attack_epsilon_start/255
            
            if args.constant_kappa:
                kappa = args.kappa_end
            else:
                kappa = (1-lin_coeff)*1 + lin_coeff*args.kappa_end

            data = replay_buffer.sample(args.batch_size)
            if args.robust:
                loss, standard_loss, worst_case_loss = _compute_robust_loss(current_model, target_model, data, attack_epsilon,
                                                                     kappa, args.gamma, device, args, args.attack_epsilon_end)
            
            else:
                raise ValueError("Not Robust Training")
                #loss, standard_loss, worst_case_loss = _compute_loss(current_model, target_model, data, args.gamma, device)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            standard_losses.append(standard_loss.data.item())
            worst_case_losses.append(worst_case_loss.data.item())

        if frame_idx % (1000*(args.batch_size/args.updates_per_frame)) == 0:
            target_model.load_state_dict(current_model.state_dict())
            
    #save final model
    state_to_save = {'model_state_dict':current_model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'step':frame_idx}
                     
    if args.resume_training:
        torch.save(state_to_save, args.resume_training[:-3]+"_last.pt")
    else:
        torch.save(state_to_save, '{}{}_{}_last.pt'.format(args.save_model_dir, args.env, start_time))
                
    log['{}_log'.format(args.env)].info("Done in {:.3f}s".format(time.time()-start))
    
    #evaluation for the final model
    #os.system("python evaluate.py --env {} --load-path \"{}{}_{}_last.pt\" --pgd --nominal --gpu-id 0 --eps {} {}".format(
    #        args.env, args.save_model_dir, args.env, start_time, args.attack_epsilon_start, args.attack_epsilon_start+1))
          
    print("{}{}_{}_last.pt".format(args.save_model_dir, args.env, start_time))    

class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done, pert_delta):
        self.buffer.append((state, action, reward, next_state, done, pert_delta))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state, action, reward, next_state, done, pert_delta = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state, dim=0), torch.cat(action, dim=0), torch.cat(reward, dim=0), 
                torch.cat(next_state, dim =0), torch.cat(done, dim=0), torch.cat(pert_delta, dim = 0))
        
    def __len__(self):
        return len(self.buffer)
        