import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = self.t < self.learning_starts or np.random.random() < eps
        if perform_random_action:
            action = np.random.randint(self.num_actions)
        else: 
            recent_obv = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(recent_obv)
        
        # TODO take a step in the environment using the action from the policy
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs

        # TODO store the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

            # TODO update the target network periodically 
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
