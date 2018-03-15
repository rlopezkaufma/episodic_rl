from environments.gridworld import GridWorld
from agents.agent import Agent
from controls.control import Sarsa, QLearning, ExpectedSarsa
from utils.train import Train
from utils.comparator import Comparator
from policies.action_value import TabularActionValue

# Train an agent in a episodic gridworld with windy tiles using Sarsa, QLearning
# and ExpectedSarsa control algorithms
trainings = []
controls = [Sarsa(), QLearning(), ExpectedSarsa()]
for control in controls:
    game = GridWorld(level=1)
    action_value = TabularActionValue(game.get_states(), game.get_actions())
    agent = Agent(game, action_value)
    train = Train(agent, game, control)
    train.train()
    trainings.append(train)

# Compare the episodic rewards obtained by the agent during training for the
# different control algorithms
comp = Comparator(*trainings, smoothing=30)
comp.compare_rewards()
