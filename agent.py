"""Agent evolving in an enviroment whose behavior policy is implemented using
its action-value function
"""

class Agent:
    """Simple agent which can learn to interact with an environment"""
    def __init__(self, environment, action_value, greedy=False):
        """Initialize the agent

        Args:
          environment: environment the agent evolves in
          action_value: action_value function determining the agent's behavior
              policy
          greedy: is the agent's behavior policy greedy?
        """
        self.environment = environment
        self.action_value = action_value
        self.greedy = greedy

    def act_once(self):
        """Make the agent perform the next action under the behavior policy"""
        action = self.action_value.choose_next_action(
            self.environment.get_agent_state(), greedy=self.greedy)
        self.environment.perform_action(action)

    def get_next_action_value(self, environment):
        """Evaluate the action-value of what would be the next action under the
        behavior policy **without** actually making the agent perform the
        action

        Args:
          environment: environment with which the agent is interacting
        """
        state = environment.get_agent_state()
        next_action = self.action_value.choose_next_action(state,
                                                           greedy=self.greedy)
        return self.action_value.get_action_value(state, next_action)

    def get_action_value_to_update(self, environment):
        """Get the action-value of the pair (state, action) we want to update
        In this case its the latest (state, action) pair the agent has
        transitioned from

        Args:
          environment: environment with which the agent is interacting
        """
        previous_state = environment.get_previous_agent_state()
        latest_action = environment.get_latest_action()
        return self.action_value.get_action_value(previous_state,
                                                  latest_action)

    def get_expected_next_action_value(self, environment):
        """Get the expected action-value of the agent's next state

        Args:
          environment: environment with which the agent is interacting
        """
        return self.action_value.get_expected_action_value(
            environment.get_agent_state())

    def get_best_next_action_value(self, environment):
        """Get the best action-value given the agent's next state

        Args:
          environment: environment with which the agent is interacting
        """
        return self.action_value.get_best_action_value(
            environment.get_agent_state())

    def update_action_value(self, value, environment):
        """Update the agent's action-value function with the output of the
        control algorithm

        Args:
          value (float): output of the control algorithm
          environment: environment with which the agent is interacting
        """
        state = environment.get_previous_agent_state()
        action = environment.get_latest_action()
        self.action_value.update_action_value(state, action, value)
