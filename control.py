"""Time difference control algorithms"""

class TDControl:
    """Action value temporal difference control
    Generalized policy iteration algorithm where the temporal difference method
    is used to estimate and improve the action-value function of the agent
    """
    def __init__(self, alpha=0.5, discount=1.0):
        """Initialize the action value temporal difference control algorithm

        Args:
          alpha (float): learning rate for the update step targeting `target`:
              Q(s,a) <- Q(s, a) - alpha*target
         discount (float): discount rate for the return G:
              G_t = R_t + discount*R_{t+1} + discount**2*R_{t+2} +...
        """
        self.alpha = alpha
        self.discount = discount

    def update(self, agent, environment):
        """Update the agent's action-value function using the feedback from
        the environment

        Args:
            agent: agent whose action-value is being learned
            environment: environment with which the agent is interacting
        """
        raise NotImplementedError()


class Sarsa(TDControl):
    """On-policy TD control algorithm using the Sarsa prediction method

    Given a 5-tuple (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}) representing the
    transition from one (state, action) pair to the next following to the
    behavior policy, maximize the action-value function of the agent under the
    behavior policy using a temporal difference method
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'sarsa'

    def update(self, agent, environment):
        """Update the agent's action-value function using the target:
        R_t + discount*Q(S_{t+1}, A_{t+1})
        """
        target = (environment.get_latest_reward()
                  + self.discount*agent.get_next_action_value(environment))
        td_error = target - agent.get_action_value_to_update(environment)
        agent.update_action_value(self.alpha*td_error, environment)


class ExpectedSarsa(TDControl):
    """On-policy TD control algorithm using the expected Sarsa prediction
    method

    Given a 4-tuple (S_t, A_t, R_{t+1}, S_{t+1}) representing the transition
    from one (state, action) pair to the next state following the behavior
    policy, maximize the action-value function of the agent under the behavior
    policy using a temporal difference method
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'expected_sarsa'

    def update(self, agent, environment):
        """Update the agent's action-value function using the target:
        R_t + discount*E(Q(S_{t+1}, A))
        """
        target = (environment.get_latest_reward()
                  + self.discount*agent.get_expected_next_action_value(
                      environment))
        td_error = target - agent.get_action_value_to_update(environment)
        agent.update_action_value(self.alpha*td_error, environment)


class QLearning(TDControl):
    """Off-policy TD control algorithm

    Given a 4-tuple (S_t, A_t, R_{t+1}, S_{t+1}) representing the transition
    from one (state, action) pair to the next state following the behavior
    policy, approximate the **optimal** action-value function of the agent
    using a temporal difference method independently of the policy being
    followed by the agent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'qlearning'

    def update(self, agent, environment):
        """Update the agent's action-value function using the target:
        R_t + discount*max_a(Q(S_{t+1}, a))
        """
        target = (environment.get_latest_reward()
                  + self.discount*agent.get_best_next_action_value(
                      environment))
        td_error = target - agent.get_action_value_to_update(environment)
        agent.update_action_value(self.alpha*td_error, environment)
