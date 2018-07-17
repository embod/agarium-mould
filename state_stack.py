import numpy as np

class StateStack:

    def __init__(self, max_states=5):
        self._state_stack = []
        self._max_states = max_states
        self._frame_skip_counter = 0

    def add_state(self, state):
        if len(self._state_stack) == self._max_states:
            del self._state_stack[0]
        self._state_stack.append(state)

    def split_state(self, state):
        sensor = state[0:50]

        xvelocity = state[50]
        yvelocity = state[51]
        radius = state[52]

        return sensor, xvelocity, yvelocity, radius

    def split_means(self):
        """
        Averages out the last max_states states
        :return:
        """
        means = np.mean(np.array(self._state_stack), axis=0)

        return self.split_state(means)

    def split_frame_skip(self):

        if self._frame_skip_counter == self._max_states:
            self._frame_skip_counter = 0
            return self.split_state(self._state_stack[-1])
        else:
            self._frame_skip_counter += 1
            return None

    def get_state(self):
        return self._state_stack[-1]