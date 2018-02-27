"""Action-value function to implement the agent's behavior policy"""
from operator import add
import numpy as np

class ActionValue:
    """Action value function `Q`, where `Q(s,a)` is the expected discounted return
    of taking action `a` in state `s`
    """
    def __init__(self, states, actions, epsilon=0.1, double=False):
        """Initialize an action-value function

        Args:
          states: states defined by the environment
          actions: actions which the agent can take in the environment
          epsilon (float): controls how greedy the policy is:
              if a U(0,1) is < epsilon, then choose an action unformly at random
              otherwise, choose the action which maximizes the action-value
        """
        self.actions = actions
        self.states = states
        self.epsilon = epsilon

        # When using TD control methods such as Qlearning where one tries to
        # approximate the optimal action-value function by solving iteratively
        # a Bellman equation there's an inherent maximization bias which occurs
        # due to the fact that the same function is used to find the greedy
        # action and to update the action-value of that greedy action.

        # This problem can be overcome by maintaining two estimates of the
        # action-value function using one to find the greedy action while using
        # the other to update its action-value
        self.double = double

        # Flag controlling which action-value function estimate to use for
        # finding the greedy action
        self.flag = 0

    def get_action_value(self, state, action):
        """Get the action-value of a given (state, action) pair

        Args:
          state: one of the defined states of the environment
          action: one of the actions which can be taken by the agent in a given
              state

        Raises:
          ValueError: if requesting the action-value of a state which is not
              one of the defined states of the environment or of an action which
              cannot be taken in a given state
        """
        raise NotImplementedError()

    def update_action_value(self, state, action, update):
        """Update the value of a given (state, action) pair

        Args:
          state: one of the defined states of the environment
          action: one of the actions which can be taken by the agent in a given
              state
          update (float): value of the update as calculated by the control
              algorithm

        Raises:
          ValueError: if updating the action-value of a state which is not
              one of the defined states of the environment or of an action which
              cannot be taken in the state
        """
        raise NotImplementedError()

    def choose_next_action(self, state, greedy=False):
        """Choose the next action under the behavior policy in a given state

        Args:
          state: one of the defined states of the environment
          greedy (bool): if true the behavior policy is greedy, otherwise it's
              epsilon-greedy

        Raises:
          ValueError: if requesting the next action in state which is not one
              of the defined states of the environment
        """
        raise NotImplementedError()

    def get_expected_action_value(self, state):
        """Get the expected action-value under the behavior policy in a given state

        Args:
          state: one of the defined states of the environment

        Raises:
          ValueError: if requesting the expected action-value of a state which
              is not one of the defined states of the environment
        """
        raise NotImplementedError()

    def get_best_action_value(self, state):
        """Get the action-value under the behavior policy of the action which
        maximizes the action-value function in a given state

        Args:
          state: one of the defined states of the environment

        Raises:
          ValueError: if requesting the maximum action-value of a state which
              is not one of the defined states of the environment
        """
        raise NotImplementedError()

    def _validate_action(self, state, action):
        """Check that the action is one of the actions which can be taken by the
        agent in a given state

        Args:
          state: one of the defined states of the environment
          action: one of the actions which can be taken by the agent in a given
              state

        Raises:
          ValueError: if requesting the action-value of an action which cannot
              be taken in the state
        """
        if action < 0 or action >= len(self.actions):
            raise ValueError("Action {0} cannot be taken in state {1}"
                             .format(action, state))


class TabularActionValue(ActionValue):
    """Tabular action-value function: finite number of (action-state) pairs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.double:
            self.name = 'double_tabular'
        else:
            self.name = 'simple_tabular'
        # The action-value function in this case is represented as a dict whose
        # keys are the states and whose values are arrays where at index `i`
        # is stored the expected discounted return of taking the ith action in
        # the keyed state
        self.tabular_action_value = {state: [0]*len(self.actions) for state \
                                     in self.states}

        # The auxiliary action-value function used to avoid maximization bias
        self.auxiliary_tabular_action_value = {state: [0]*len(self.actions) \
                                               for state in self.states}

    def _validate_state(self, state):
        """Check that the state is a defined state of the environment

        Args:
          state: one of the defined states of the environment

        Raises:
          ValueError: if requesting a state which is not one of the defined
              states of the environment
        """
        if not state in self.tabular_action_value:
            raise ValueError("State {0} is not defined in the environment")

    def get_action_value(self, state, action):
        self._validate_state(state)
        self._validate_action(state, action)

        if not self.double or self.flag == 0:
            value = self.tabular_action_value[state][action]
        else:
            value = self.auxiliary_tabular_action_value[state][action]
        return value

    def update_action_value(self, state, action, update):
        self._validate_state(state)
        self._validate_action(state, action)

        if not self.double or self.flag == 0:
            self.tabular_action_value[state][action] += update
            self.flag = 1
        else:
            self.auxiliary_tabular_action_value[state][action] += update
            self.flag = 0

    def choose_next_action(self, state, greedy=False):
        self._validate_state(state)
        if not greedy and np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            if self.double:
                # This inner map is potentially executed on each iteration of
                # the learning phase. However, most of the time, it's a constant
                # time operation since in most games there are usually less than
                # a few tens possible moves
                avg_action_value = map(add, self.tabular_action_value[state],
                                       self.auxiliary_tabular_action_value[state])
                action = np.argmax(list(avg_action_value))
            else:
                action = np.argmax(self.tabular_action_value[state])
        return action

    def get_expected_action_value(self, state):
        self._validate_state(state)

        if self.double and self.flag == 0:
            action_value_function = self.auxiliary_tabular_action_value
        else:
            action_value_function = self.tabular_action_value

        best_action = np.argmax(action_value_function[state])
        expected_next_action_value = 0
        for action, action_value in enumerate(action_value_function[state]):
            if action == best_action:
                multiplier = 1-self.epsilon*(1-1.0/len(self.actions))
                expected_next_action_value += action_value*multiplier
            else:
                multiplier = self.epsilon*1.0/len(self.actions)
                expected_next_action_value += action_value*multiplier
        return expected_next_action_value

    def get_best_action_value(self, state):
        self._validate_state(state)

        if self.double and self.flag == 0:
            value = np.max(self.auxiliary_tabular_action_value[state])
        else:
            value = np.max(self.tabular_action_value[state])
        return value
