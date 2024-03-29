import numpy as np
import os
import torch.optim as optim
from memoryBuffers import ReplayBuffer, ReplayBuffer2, Buffer
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from networks import CategoricalPolicy

class Player():
  """
  Description:  A class to describe the player.  This class stores relavant
  player information such as:
    1.  Player Color
    2.  Population Group
    4.  Defection History
    5.  Experience Replay Memory
    6.  Player ID

  This class will also provide a way to update player policies.
  """
  def __init__(self,
                   color,
                   population,
                   obs_dim,
                   act_dim,
                   memory=1,
                   act_limit=1.0,
                   device="cpu",
                   player_id=None,
                   configure_save_path=True,
                   save_path='/'):

        # generate random player id if not passed
        if player_id == None:
          self.player_id = np.random.randint(0, 1000000)
        else:
          self.player_id = player_id

        self.memory = memory
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.color = color
        self.population = population
        self.device = device
        self.save_path = save_path

        if configure_save_path:
            self.configure_save_path()

  def configure_save_path(self):
       for player_color in ['/r', '/b']:
           if not os.path.exists(self.save_path + player_color):
               os.makedirs(self.save_path + player_color)
               print(f"Directory '{self.save_path + player_color}' created.")
           else:
               print(f"Directory '{self.save_path + player_color}' already exists.")
               pass

           if not os.path.exists(self.save_path + player_color + f'/{self.population}'):
               os.makedirs(self.save_path + player_color + f'/{self.population}')
               print(f"Directory '{self.save_path + player_color + f'/{self.population}'}' created.")
           else:
               print(f"Directory '{self.save_path + player_color + f'/{self.population}'}' already exists.")

       return 1


class DQNPlayer(Player):
    def __init__(self, 
                base_player_params,
                model,
                model_params,
                network='dqn',
                steps=0,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                buffer_size=int(1e4),
                batch_size=64,
                target_update_step=100,
                q_losses=list() ):
      
      super().__init__(**base_player_params)

      self.network=network
      self.steps = steps
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_decay = epsilon_decay
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.target_update_step = target_update_step
      self.q_losses = q_losses

      # Get the current date

      # Main network
      self.qf = model(**model_params).to(self.device)

      # Target network
      self.qf_target = model(**model_params).to(self.device)
      
      # Initialize target parameters to match main parameters
      self.qf_target.load_state_dict(self.qf.state_dict())

      # Create an optimizer
      self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)

      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, 1, self.buffer_size, self.device)


    def add_experience(self, obs, action, reward, next_obs, done):
     self.replay_buffer.add(obs, action, reward, next_obs, done)


    def select_action(self, obs, eval_mode=False):
        """Select an action from the set of available actions."""
        # Decaying epsilon

        if eval_mode:
          action = self.qf(obs).argmax()
          return action.detach().cpu().item()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)

        if np.random.rand() <= self.epsilon:
          # Choose a random action with probability epsilon
          return np.random.randint(self.act_dim)
        else:
          # Choose the action with highest Q-value at the current state
          action = self.qf(obs).argmax()
          return action.detach().cpu().item()
   
    def save_network(self):
        torch.save(self.qf.state_dict(), f'{self.save_path}/{self.color}/{self.population}/DQN_Player_{self.player_id}_VF_SD.csv')
        return 1

    def save_policy(self, df2, cols=['A1', 'A2', 'A3', 'A4']):
      df = df2.copy()

      df[cols]  = df.apply(lambda row: self.qf(torch.Tensor(row.values)).detach().numpy(), axis=1, result_type='expand')
      df.to_csv(f'{self.save_path}/{self.color}/{self.population}/DQN_Player_{self.player_id}_VF.csv')

      print("Player Info Saved")
      return 1

    def train_model(self):
        # only train if we have more steps than batchsize
        if self.steps < self.batch_size:
          return

        batch = self.replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        # Prediction Q(s)
        q = self.qf(obs1).gather(1, acts.long()).squeeze(1)
        
        # Target for Q regression
        if self.network == 'dqn':      # DQN
          q_target = self.qf_target(obs2)

        elif self.network == 'ddqn':   # Double DQN
          q2 = self.qf(obs2)
          q_target = self.qf_target(obs2)
          q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))

        q_backup = rews + self.gamma*(1-done)*q_target.max(1)[0]
        q_backup.to(self.device)


        # Update perdiction network parameter
        qf_loss = F.mse_loss(q, q_backup.detach())
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Synchronize target parameters 𝜃‾ as 𝜃 every C steps
        if self.steps % self.target_update_step == 0:
          # Initialize target parameters to match main parameters
          self.qf_target.load_state_dict(self.qf.state_dict())
        
        # Save loss
        self.q_losses.append(qf_loss.item())


class PPOPlayer(Player):
   """
   An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent, 
   with early stopping based on approximate KL.
   """
   def __init__(self, 
                base_player_params,
                actor_model,
                actor_model_params,
                critic_model,
                critic_model_params,
                training_params):

      # initialize player object params
      super().__init__(**base_player_params)

      # training params
      self.steps = training_params.get('steps', 0)
      self.gamma = training_params.get('gamma', 0.99)
      self.lam = training_params.get('lam', 0.97)
      self.sample_size = training_params.get('sample_size', 2048)
      self.buffer_multiple = training_params.get('buffer_multiple', 2)
      self.train_policy_iters = training_params.get('train_policy_iters', 10)
      self.train_vf_iters = training_params.get('train_vf_iters', 10)
      self.clip_param = training_params.get('clip_param', 10)
      self.target_kl = training_params.get('target_kl', 10)
      self.policy_lr = training_params.get('policy_lr', 3e-4)
      self.vf_lr = training_params.get('vf_lr', 3e-4)
      self.policy_losses = training_params.get('policy_losses', list())
      self.vf_losses = training_params.get('vf_losses', list())
      self.kls = training_params.get('kls', list())

      # Main network
      self.policy = actor_model(**actor_model_params).to(self.device)
      self.vf = critic_model(**critic_model_params).to(self.device)
      
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size*self.buffer_multiple, self.device, self.gamma, self.lam)


   def select_action(self, obs, eval_mode=False):
      if eval_mode == False:
         # actions sampled from distribution means and stds of each
         # action. Then standardized to prob dist
         action, _, _, _ = self.policy(torch.Tensor(obs).to(self.device))
         return action.detach().cpu().numpy().flatten()[0]
      else:
        # action is the largus mu from our guassian policy
        _, _, action, _ = self.policy(torch.Tensor(obs).to(self.device))

        action = action.detach().cpu().numpy().flatten()

        # softmax to convert to provs
        return np.random.choice(range(0, self.act_dim), p=np.exp(action)/sum(np.exp(action)))


   def compute_vf_loss(self, obs, ret, v_old):
      # Prediction V(s)
      v = self.vf(obs).squeeze(1)

      # Value loss
      clip_v = v_old + torch.clamp(v-v_old, -self.clip_param, self.clip_param)
      vf_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(clip_v, ret)).mean()
      return vf_loss


   def compute_policy_loss(self, obs, act, adv, log_pi_old):
      # Prediction logπ(s)
      _, _, _, log_pi = self.policy(obs, act, use_pi=False)
      
      # Policy loss
      ratio = torch.exp(log_pi - log_pi_old)
      clip_adv = torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)*adv
      policy_loss = -(torch.min(ratio*adv, clip_adv)).mean()

      # A sample estimate for KL-divergence, easy to compute
      approx_kl = (log_pi_old - log_pi).mean()
      return policy_loss, approx_kl


   def train_model(self):
      # Start training when the number of experience is equal to sample size
      if self.steps == self.sample_size:
        self.buffer.finish_path()
        self.steps = 0
      else:
        # don't train
        return

      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      ret = batch['ret']
      adv = batch['adv']
      
      # Prediction logπ_old(s), V_old(s)
      _, _, _, log_pi_old = self.policy(obs, act, use_pi=False)
      log_pi_old = log_pi_old.detach()
      v_old = self.vf(obs).squeeze(1)
      v_old = v_old.detach()

      # Train policy with multiple steps of gradient descent
      for i in range(self.train_policy_iters):
         policy_loss, kl = self.compute_policy_loss(obs, act, adv, log_pi_old)

         # Early stopping at step i due to reaching max kl
         if kl > 1.5 * self.target_kl:
            break

         # Update policy network parameter
         self.policy_optimizer.zero_grad()
         policy_loss.backward()
         self.policy_optimizer.step()

      # Train value with multiple steps of gradient descent
      for i in range(self.train_vf_iters):
         vf_loss = self.compute_vf_loss(obs, ret, v_old)

         # Update value network parameter
         self.vf_optimizer.zero_grad()
         vf_loss.backward()
         self.vf_optimizer.step()

      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.kls.append(kl.item())


   def add_experience(self, obs, action, reward, next_state, done):

       # critic estimate
       v = self.vf(torch.Tensor(obs).to(self.device))
       self.buffer.add(obs, action, reward, done, v)
       return 1

   def save_network(self):
       torch.save(self.policy.state_dict(), f'{self.save_path}/{self.color}/{self.population}/PPO_Player_{self.player_id}_PN_SD.csv')
       torch.save(self.vf.state_dict(), f'{self.save_path}/{self.color}/{self.population}/PPO_Player_{self.player_id}_VF_SD.csv')
       return 1

   def save_policy(self, df2, cols=['A1', 'A2', 'A3', 'A4']):
      df = df2.copy()
      df[cols] = df.apply(lambda row: self.policy(torch.Tensor(row))[2].detach().numpy(), axis=1, result_type='expand')
      df[cols] = np.exp(df[cols])
      df[cols] = df[cols]/np.sum(df[cols], axis=0)
      df.to_csv(f'{self.save_path}/{self.color}/{self.population}/PPO_Player_{self.player_id}_PN.csv')
      print("Player Info Saved")

      return 1


class VPGPlayer(Player):
    """
    An implementation of the Vanilla Policy Gradient agent,
    with early stopping based on approximate KL.
    """

    def __init__(self,
                 base_player_params,
                 actor_model,
                 actor_model_params,
                 critic_model,
                 critic_model_params,
                 training_params):

        # initialize player object params
        super().__init__(**base_player_params)

        # training params
        self.steps = training_params.get('steps', 0)
        self.gamma = training_params.get('gamma', 0.99)
        self.lam = training_params.get('lam', 0.97)
        self.sample_size = training_params.get('sample_size', 2048)
        self.buffer_multiple = training_params.get('buffer_multiple', 2)
        self.train_vf_iters = training_params.get('train_vf_iters', 10)
        self.policy_lr = training_params.get('policy_lr', 3e-4)
        self.vf_lr = training_params.get('vf_lr', 3e-4)
        self.policy_losses = training_params.get('policy_losses', list())
        self.vf_losses = training_params.get('vf_losses', list())
        self.kls = training_params.get('kls', list())

        # Main network
        self.policy = actor_model(**actor_model_params).to(self.device)
        self.vf = critic_model(**critic_model_params).to(self.device)

        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)

        # Experience buffer
        self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size*self.buffer_multiple, self.device, self.gamma, self.lam)

    def select_action(self, obs):
        action, _, _, _ = self.policy(torch.Tensor(obs).to(self.device))
        return action.detach().cpu().numpy().flatten()[0]

    def train_model(self):
        # Start training when the number of experience is equal to sample size
        if self.steps == self.sample_size:
            self.buffer.finish_path()
            self.steps = 0
        else:
            # don't train
            return

        batch = self.buffer.get()
        obs = batch['obs']
        act = batch['act'].detach()
        ret = batch['ret']
        adv = batch['adv']

        for _ in range(self.train_vf_iters):
            # Prediction V(s)
            v = self.vf(obs).squeeze(1)
            vf_loss = F.mse_loss(v, ret)
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        # Prediction logπ(s)
        _, _, _, log_pi_old = self.policy(obs, act, use_pi=False)
        log_pi_old = log_pi_old.detach()
        _, _, _, log_pi = self.policy(obs, act, use_pi=False)

        # Policy loss
        policy_loss = -(log_pi * adv).mean()

        # Update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # A sample estimate for KL-divergence, easy to compute
        approx_kl = (log_pi_old - log_pi).mean()

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.vf_losses.append(vf_loss.item())
        self.kls.append(approx_kl.item())

    def add_experience(self, obs, action, reward, next_state, done):
        # critic estimate
        v = self.vf(torch.Tensor(obs).to(self.device))
        self.buffer.add(obs, action, reward, done, v)
        return 1

    def save_network(self):
        torch.save(self.policy.state_dict(),
                   f'{self.save_path}/{self.color}/{self.population}/VPG_Player_{self.player_id}_PN_SD.csv')
        torch.save(self.vf.state_dict(),
                   f'{self.save_path}/{self.color}/{self.population}/VPG_Player_{self.player_id}_VF_SD.csv')
        return 1

    def save_policy(self, df2, cols=['A1', 'A2', 'A3', 'A4']):
        df = df2.copy()
        df[cols] = df.apply(lambda row: self.policy(torch.Tensor(row))[2].detach().numpy(), axis=1,
                            result_type='expand')
        df[cols] = np.exp(df[cols])
        df[cols] = df[cols] / np.sum(df[cols], axis=0)
        df.to_csv(f'{self.save_path}/{self.color}/{self.population}/VPG_Player_{self.player_id}_PN.csv')
        print("Player Info Saved")

        return 1


class VPGPlayer2(Player):
    """
    An implementation of the Vanilla Policy Gradient agent,
    with early stopping based on approximate KL.
    """

    def __init__(self,
                 base_player_params,
                 actor_model_params,
                 training_params
                 ):

        # initialize player object params
        super().__init__(**base_player_params)

        self.steps = 0

        self.buffer_size = training_params.get('buffer_size', 300)
        self.batch_size = training_params.get('batch_size', 64)
        self.policy_lr = training_params.get('policy_lr', 0.01)
        self.policy_target = training_params.get('policy_target', 0)

        # Main network
        self.policy = CategoricalPolicy(**actor_model_params).to(self.device)

        # Create optimizers
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        # Experience buffer
        self.buffer = ReplayBuffer2(self.obs_dim, self.act_dim, self.buffer_size, self.device)

    def select_action(self, obs):
        action, _, probs, log_pi = self.policy(torch.Tensor(obs).to(self.device))
        return action.detach().cpu().numpy().flatten()[0]

    def train_model(self, get_all=0):
        if get_all == 1:
            batch = self.buffer.get_all()
        else:
            batch = self.buffer.get(batch_size=self.batch_size)
        states = batch['obs']
        actions = batch['acts']
        rewards = batch['rews']

        total_reward = torch.sum(rewards)
        policy_target = total_reward - self.policy_target*(len(rewards))

        # get probabilities from policy net
        _, _, probs, _ = self.policy(states)

        # get probs/log probs of each action
        probs = Categorical(probs)
        log_probs = probs.log_prob(actions)

        # Total reward formulation
        loss = torch.sum(-log_probs * policy_target)

        # update policy weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def add_experience(self, obs, action, reward, next_state, done):
        self.buffer.add(obs, action, reward)
        return 1

    def save_network(self):
        torch.save(self.policy.state_dict(),
                   f'{self.save_path}/{self.color}/{self.population}/VPG_Player_{self.player_id}_PN_SD.csv')
        return 1

    def save_policy(self, df2, cols=['A1', 'A2', 'A3', 'A4']):
        df = df2.copy()
        df[cols] = df.apply(lambda row: self.policy(torch.Tensor(row))[2].detach().numpy(), axis=1,
                            result_type='expand')
        df[cols] = np.exp(df[cols])
        df[cols] = df[cols] / np.sum(df[cols], axis=0)
        df.to_csv(f'{self.save_path}/{self.color}/{self.population}/VPG_Player_{self.player_id}_PN.csv')
        print("Player Info Saved")

        return 1