"""Compare control algorithms"""

import matplotlib.pyplot as plt

class Comparator:
    """Compare multiple generalized policy iteration methods used to train the
    same agent in the same environment for the same number of episodes
    """
    def __init__(self, *trainings, smoothing=10):
        """Initialize comparator

        Args:
            trainings: list of Train instances
            smoothing: parameter controlling how smooth the graphs should be
        """
        self.trainings = trainings
        self.smoothing = smoothing

    def _smooth(self, timeseries):
        """Smooth a timeseries by using a moving average with a self.smooth
        time window
        """
        smoothed_ts = []
        buffer = timeseries[0:self.smoothing]
        running_sum = 0
        for idx, datapoint in enumerate(timeseries):
            if idx < self.smoothing:
                smoothed_ts.append(datapoint)
                running_sum += datapoint
            else:
                running_sum = running_sum - buffer[0] + datapoint
                smoothed_ts.append(running_sum/self.smoothing)
                del buffer[0]
                buffer.append(datapoint)
        return smoothed_ts

    def _legend(self, training):
        return "-".join([training.control.name,
                         training.agent.action_value.name])

    def compare_rewards(self):
        """Plot the cumulative rewards obtained during training by each of the
        different control algorithms
        """
        for training in self.trainings:
            smoothed_rewards = self._smooth(training.episode_rewards)
            plt.plot(smoothed_rewards)

        plt.legend([self._legend(training) for training in self.trainings],
                   loc='upper left')
        plt.show()
