import flappy_bird_gymnasium
import gymnasium as gym
import torch 
from experience_replay import ReplayMemory
import itertools
import torch.nn as nn 
import torch.optim as optim
import yaml
from dqn import DQN
import random
import os 
import argparse

#device
if torch.backends.is_available():
    device="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
    
#Creating Directory
RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
class Agent():
    def __init__(self,param_set):
        self.param_set=param_set
        
        with open("parameters.yaml","r") as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]
            
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.reward_threshold = params["reward_threshold"]
        self.loss_fn=nn.MSELoss()
        self.optimizer=None
        
        
        #Adding Log Files
        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_LIFE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")
        
        
    def run(self,is_training=True,render=False):
        env=gym.make("FlappyBird-v0",render_mode="human" if render else None)
        num_states=env.observation_space.shape[0]  #Input dim
        num_actions=env.action_space.n  #output dim
        
        policy_dqn=DQN(num_states,num_actions).to(device)
        
        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
            
            
            #Target DQN
            target_dqn=DQN(num_states,num_actions).to(device)
            #copy the bias and wt from this target network
            target_dqn.load_state_dict(policy_dqn.state_dict())    #state_dict() gives bias and weight
            
            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            
            #Best reward only needs to store in the RUNS_DIR
            best_reward=float("inf")
        
        else:   #Testing Case we want to load best model
            policy_dqn.load_state_dict(torch.load(self.MODEL_LIFE))
            policy_dqn.eval()
            
            
        for episode in itertools.count():
            state, _ = env.reset()
            
            # We will feed the state, action and reward to the neural network
            state=torch.tensor(state,dtype=torch.float,device=device)
            
            episode_reward=0
            terminated=False      
                
            while (not terminated and episode_reward< self.reward_threshold):  #Chalte rhanea hai jab tak episode reward < reward_threshold
                if is_training and random.random()<epsilon:
                    action=env.action_space.sample()
                    action=torch.tensor(state,dtype=torch.float,device=device)
                else:
                    with torch.no_grad():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()   #Exploit when not training-> Testing
                    #Add a dim at idx 0 
                    #DQN expects 2D we have 1D 
                # Next action:
                # (feed the observation to your agent here)
                    

                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item())  #Converting back to list from tensor
                
                #reward ko tensor me dala jayega isliye phele hi kar diya simple way me 
                episode_reward+=reward
                
                reward=torch.tensor(reward,dtype=torch.float,device=device)
                next_state=torch.tensor(next_state,dtype=torch.float,device=device)
                
                if is_training:
                        memory.append((state,action,next_state,reward,terminated))
                        steps+=1
                        
                state=next_state
                
            print("print epsilon")
            # env.close()
            
            if is_training:
            #epsilon decay calc
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
                if episode_reward > best_reward:
                    log_msg=f"best_reward={best_reward} for episode ={episode+1}"
                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg+"\n")
                        
                    torch.save(policy_dqn.state_dict(),self.MODEL_LIFE)
                    best_reward=episode_reward
                
            if is_training and len(memory)>self.mini_batch_size:
                #get sample
                mini_batch=memory.sample(self.mini_batch_size)
                #Mini batch ka sample chaiye
                self.optimize(mini_batch,policy_dqn,target_dqn)
                
                #Agar hamari steps jyada ho jate hai nw sync rate se Then 
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())  
                    steps=0 #To reset  
                    
    # def optimize(self,mini_batch,policy_dqn,target_dqn):
    #     #get all experiences first
    #     for state,action,next_state,reward,terminated in mini_batch:
    #         if terminated:
    #             target=reward
    #         else:
    #             with torch.no_grad():
    #                 target_q=reward+self.gamma*target_dqn(next_state).max()  #True y
                                    
    #         current_q=policy_dqn(state)
    #         #Ek ek karke process kar rhe hai hum but slow so train in batch for that
    #         #cal loss
            
    #         loss=self.loss_fn(current_q,target_q)
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()  #Update wt + bias also 
            
            
            
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        #get all experiences first
        states,actions,next_states,rewards,terminations = zip(*mini_batch)   #Zip unpacks the fn
        
        states=torch.stack(states)
        actions=torch.tensor(actions,dtype=torch.long,device=device)
        mini_batch=torch.stack(mini_batch)
        rewards=torch.tensor(rewards,dtype=torch.float32,device=device)
        terminations=torch.tensor(terminations,dtype=torch.bool,device=device)
    
        #Current Q values                            
        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            #Phele if terminate=0 then target =rewards ho rha tha ab direct termination 1 hua to all 0---> will give rewrads
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
            
        loss=self.loss_fn(current_q,target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  #Update wt + bias also 