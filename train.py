"""Train agent's action-value function given an environment and a control
method
"""

class Train:
    """Train the agent's policy to maximize its reward in an episodic
    environment
    """
    def __init__(self, agent, environment, control, episodes=1000):
        """Initialize the training algorithm

        Args:
          agent: agent evolving in the environment
          environment: environment in which the agent evolves
          control: generalized policy iteration algorithm
          episodes: how many episodes should we train the agent on?
        """
        self.agent = agent
        self.environment = environment
        self.control = control
        self.episodes = episodes
        # Keep track of the per episode return for diagnostic purposes
        self.episode_rewards = []

    def train(self):
        """Run the agent in the environment for each episode and improve its
        policy
        """
        for _ in range(self.episodes):
            episode_reward = self.run_episode()
            self.episode_rewards.append(episode_reward)

    def run_episode(self):
        """Run one episode of the environment"""
        self.environment.reset()
        while not self.environment.is_episode_over():
            self.agent.act_once()
            self.control.update(self.agent, self.environment)
        return self.environment.get_episode_reward()
