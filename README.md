# Very simple framework to implement and test generalized iteration policy algorithms

This very simple framework leverages the Pycolab library to help you setup
gridworld environments to test your control algorithms.
It's made of the following classes:
* GridWorld: episodic gridworld environment backed by Pycolab's engine. You can
design games by givin their ascii art representation
* TDControl: time difference control algorithm, currently Sarsa, QLearning and
ExpectedSarsa (along with their double version, to mitigate maximization bias)
are implemented
* ActionValue: action-value function that an agent can use to implement its
behavior policy and which can be trained using TDControl
* Agent: agent interacting with the GridWorld environment and following the
behavior policy defined by ActionValue
* Train: train an agent's action-value function to maximize its discounted
expected return in a GridWorld
* Comparator: helper to compare control aglorithms, currently draws on the
same plot the per episode rewards obtained by the agent during the training
phase

As an example of how the code works, you can compare Sarsa, QLearning and
ExpectedSarsa on a simple gridworld with windy tiles by running
`python simple_example_usage.py`
