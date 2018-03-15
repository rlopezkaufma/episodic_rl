"""Episodic gridworld environment"""

import curses
import numpy as np
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites

class GridWorld:
    """A simple gridworld with walls and windy tiles where the goal is to reach
    a special tile based on the Pycolab game engine

    Please refer to Pycolab's github project for an introduction on how the
    game engine works
    """
    # Schematic representation of the gridworld:
    #    -'A' represents the agent
    #    -'G' represents the goal tile which the agent should read to win
    #    -'#' impassable wall
    #    -'.' windy tile which cahses the agent to move one extra tile to the
    #    west
    #    -'+' windy tile which causes the agent to move two extra tiles to the
    #    west
    LEVELS = [
        # Level 0: single wall and no wind
        {'ascii_grid': ['####################',
                        '#                 G#',
                        '#                  #',
                        '#                  #',
                        '#                  #',
                        '#                  #',
                        '#                  #',
                        '################## #',
                        '#                  #',
                        '#A                 #',
                        '####################'],
         'nrows': 11,
         'ncols': 20,
         'actions': [0, 1, 2, 3]},
        # Level 1: single wall and sideways windy tiles
        {'ascii_grid': ['####################',
                        '#G                 #',
                        '#                  #',
                        '#..................#',
                        '#++++++++++++++++++#',
                        '#..................#',
                        '#                  #',
                        '################## #',
                        '#                  #',
                        '#A                 #',
                        '####################'],
         'nrows': 11,
         'ncols': 20,
         'actions': [0, 1, 2, 3]}
    ]

    # Define the colors used to render the elemets of the gridworld to play
    # the game in the terminal
    FG_COLOURS = {
        'A': (999, 500, 0),
        '#': (700, 700, 700),
        'G': (999, 0, 0),
        ' ': (200, 200, 200),
        '.': (400, 400, 600),
        '+': (400, 400, 800)
    }
    BG_COLOURS = {
        'A': (200, 200, 200),
        '#': (800, 800, 800),
        'G': (999, 800, 800),
        ' ': (200, 200, 200),
        '.': (400, 400, 600),
        '+': (400, 400, 800)
    }

    # Define a integer mapping for the ascii art to use to define the raw
    # state
    ASCII_TO_INT = {
        ' ': 0,
        '#': 1,
        'A': 2,
        'G': 3,
        '.': 4,
        '+': 5
    }

    class Player(prefab_sprites.MazeWalker):
        """The player

        The MazeWalker class is from Pycolab and handles basic movement and
        collision detection. This class handles stopping the game when the time
        is out and detecting when the goal tile is reached.
        It might seem unnatural that the player should decide when to terminate
        the game, instead of the game engine itself, but I'm following the
        convention used in the Pycolab examples
        """
        def __init__(self, corner, position, character):
            super().__init__(corner, position, character, impassable='#')
            # How many moves has the player made so far? We need to start at -1
            # because initializing the game counts as one move
            self.elapsed = -1
            self.max_moves = None

        def set_timeout(self, timeout=60):
            """Set a maximum number of moves before terminating the episode"""
            self.max_moves = timeout

        def update(self, actions, board, layers, backdrop, things, the_plot):
            """Translate agent input into moves in the gridlword and handle
            game termination
            """
            if actions == 0:
                self._north(board, the_plot)
            elif actions == 1:
                self._south(board, the_plot)
            elif actions == 2:
                self._west(board, the_plot)
            elif actions == 3:
                self._east(board, the_plot)
            elif actions == 9:
                the_plot.terminate_episode()

            # Handle walking into a windy tile if the game has such tiles
            if '.' in layers and layers['.'][self.position]:
                # Move westward by one if possible
                self._west(board, the_plot)
            elif '+' in layers and layers['+'][self.position]:
                # Move westward by two if possibe:
                self._west(board, the_plot)
                self._west(board, the_plot)

            # Did we walk onto the goal tile? If so, terminate the game
            if layers['G'][self.position]:
                the_plot.add_reward(0)
                the_plot.terminate_episode()
            # Give a -1 reward for every move on a tile which is not the goal tile
            # to encourage the agent to reach it as fast as possible
            else:
                the_plot.add_reward(-1)

            # Add 1 to the elapsed time
            self.elapsed += 1
            # If the player has made the maximum number of moves allowed per
            # episode terminate the game.
            # We could have used the current total reward to count the number of
            # moves, but if we want to add special reward tiles it's better to
            # separate move count and total reward
            if self.elapsed == self.max_moves:
                the_plot.terminate_episode()

    def __init__(self, level=0, mode='coordinate', random_starts=False,
                 random_ops=False, timeout=60, terminal=False):
        """Initialize the gridworld

        Args:
          level (int): choose one of the preset gridworld layour
          mode: control how the agent state is defined, if set to 'coordinate'
              the state is defined as a pair of coordinate (x,y), and if set
              to 'raw' the state is defined a a greyscale image of the
              gridworld
          random_stats: if true, the starting position of the agent is drawn
              uniformly at random for the valid starting positions
          random_ops: if true a random number of no-op moves is added to the
              rules of the gridworld
          timeout (int): controls after how many moves the game is considered
              over if the goal tile hasn't been reached so far
          terminal (bool): is the game played on the terminal?
       """
        self.level = level
        self.mode = mode
        self.random_stats = random_starts
        self.random_ops = random_ops
        self.terminal = terminal
        self.timeout = timeout

        # Internal state used to communicate with the agent and the control
        # algorithm
        self.game = None
        self.latest_reward = None
        self.episode_reward = None
        self.start_state = None
        self.raw_ascii = None
        self.current_raw_ascii = None
        self.agent_actions = None
        self.agent_states = None

    def reset(self):
        """Wrapper around 'initialize' and 'start' to setup a new episode"""
        self.initialize()
        self.start()

    def initialize(self):
        """Create a new instance of the Pycolab's game engine and initialize
        the game internal state
        """
        # Instantiate the Pycolab's game engine
        self.game = ascii_art.ascii_art_to_game(
            art=GridWorld.LEVELS[self.level]['ascii_grid'],
            what_lies_beneath=' ',
            sprites={'A': GridWorld.Player},
            update_schedule=[['A']],
            z_order=['A'])
        # Set the timeout so that the player doesn't keep wandering forever
        # in the gridworld
        self.game._sprites_and_drapes['A'].set_timeout(self.timeout)

        # 2D representation of the gridworld
        self.raw_ascii = [list(row) for row in \
            GridWorld.LEVELS[self.level]['ascii_grid']]
        # That variable is updated at each timestep to reflect the position of
        # the agent in the gridworld and can be used to debug purposes
        self.current_raw_ascii = self.raw_ascii

        # Where did the player start in the gridworld
        if self.mode == 'coordinate':
            self.start_state = self.game._sprites_and_drapes['A'].position
        else:
            self.start_state = self._ascii_to_state()

        # Keep track of the total reward, of the player's actions and states
        # so far for diagnostic purposes
        self.episode_reward = 0
        self.agent_actions = []
        self.agent_states = [self.start_state]

    def start(self):
        """Freeze the gridworld configuration and enter play mode"""
        if self.terminal:
            terminal_ui = human_ui.CursesUi(
                keys_to_actions={
                    # Map keystrokes to actions
                    curses.KEY_UP: 0,
                    curses.KEY_DOWN: 1,
                    curses.KEY_LEFT: 2,
                    curses.KEY_RIGHT: 3,
                    -1: 4,  # Do nothing
                    # Quit game
                    'q': 9,
                    'Q': 9
                },
                delay=1000, # 1 second is one move
                colour_fg=GridWorld.FG_COLOURS,
                colour_bg=GridWorld.BG_COLOURS)
            terminal_ui.play(self.game)
        else:
            _, reward, _ = self.game.its_showtime()
            self.agent_states.append(self.get_agent_state())
            self.latest_reward = reward

    def _ascii_to_state(self):
        """Transforms the current ascii representation of the game state to
        a standardized array of floats
        """
        raw_state = [list(map(lambda x: GridWorld.ASCII_TO_INT[x], row) for \
                     row in self.current_raw_ascii)]
        return (raw_state-np.mean(raw_state))/np.std(raw_state)

    def get_agent_state(self):
        """Where is the player in the gridworld"""
        if self.mode == 'coordinate':
            position = self.game._sprites_and_drapes['A'].position
            state = tuple(position)
        else:
            state = self._ascii_to_state()
        return state

    def get_previous_agent_state(self):
        """Where was the agent one timestep before now?"""

        # If the agent has just performed one action or less so far it means
        # one step before it was in the start state
        if len(self.agent_states) <= 1:
            state = self.start_state
        else:
            state = self.agent_states[-2]
        return state

    def is_episode_over(self):
        """Is the episode over?"""
        return self.game.game_over

    def perform_action(self, action):
        """Execute the action chosen by the agent and modify the game internal
        state accordingly

        Args:
          action: the action received from the agent
        """
        # We need the previous state to update the ascii representation of the
        # gridworld after the action is taken
        previous_x, previous_y = self.get_agent_state()

        # Make the game engine update its state according to the action taken
        # by the player
        _, reward, _ = self.game.play(action)
        current_x, current_y = self.get_agent_state()

        # Update the extra external state the gridworld instance maintains
        self.agent_actions.append(action)
        self.agent_states.append(self.get_agent_state())
        self.episode_reward += reward

        # Update the raw ascii representation of the gridworld
        self.current_raw_ascii[previous_x][previous_y] = \
            self.raw_ascii[previous_x][previous_y]
        self.current_raw_ascii[current_x][current_y] = 'A'

        # Store the last reward incurred by the action to the agent
        self.latest_reward = reward

    def draw_play(self):
        """Draw as ascii art the path the agent has taken in the episode"""
        # Start from the ascii representation as it was at the start of the
        # episode
        raw_ascii = self.raw_ascii

        # Draw each action the agent took during the episode
        for state in self.agent_states:
            coord_y, coord_x = state
            raw_ascii[coord_y][coord_x] = '*'
        return "\n".join(map(lambda x: ''.join(x), raw_ascii))

    def get_states(self):
        """What are the defined states of the gridworld?"""
        nrows = GridWorld.LEVELS[self.level]['nrows']
        ncols = GridWorld.LEVELS[self.level]['ncols']
        if self.mode == 'coordinate':
            states = [(i, j) for i in range(nrows) for j in range(ncols)]
        else:
            states = np.zeros((nrows, ncols))
        return states

    def get_actions(self):
        """What are the actions the player can take in the gridworld?"""
        return GridWorld.LEVELS[self.level]['actions']

    def get_episode_reward(self):
        """Get the current total reward gathered by the agent during the
        episode
        """
        return self.episode_reward

    def get_latest_reward(self):
        """Get the latest reward incurred by the agent"""
        return self.latest_reward

    def get_latest_action(self):
        """What was the latest action the agent performed?"""
        return self.agent_actions[-1]
