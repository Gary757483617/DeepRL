import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None,:]

#         observation = ptu.from_numpy(observation.astype(np.float32))
        action = self(observation)
        if self.discrete:
            action_probs = nn.functional.log_softmax(action).exp()
            action_prob = torch.multinomial(action_probs, num_samples = 1)[0]
        else:
            action_prob = torch.normal(action[0], action[1])
        
        return ptu.to_numpy(action_prob).squeeze(0)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError


    def forward(self, observation: torch.FloatTensor):
        observation = ptu.from_numpy(observation.astype(np.float32))
        if self.discrete:
            action_distribution = self.logits_na(observation)
            
            return action_distribution
        else:
            action_distribution = self.mean_net(observation)
            
            return action_distribution, self.logstd.exp()


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        # TODO: update the policy and return the loss
        action_distri = self.forward(observations)
        if self.discrete:
            actions = ptu.from_numpy(actions).view(-1)
            tmp = nn.functional.log_softmax(action_distri).exp()
            action_prob = distributions.Categorical(tmp).log_prob(actions)
        else:
            action_prob = distributions.Normal(action_distri[0], action_distri[1]).log_prob(actions).sum(-1)
        
        self.optimizer.zero_grad()
        loss = torch.sum(-action_prob * ptu.from_numpy(adv_n))
        print("MLPPolicyAC loss: {}".format(loss))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
