import numpy as np
import time


class LTMatrix:
    def __init__(self, vec):
        # vec default is the acf
        self._vec = vec
        self.length = len(self._vec)
        self.lower_triang = [self._vec]
        self.filled = False

    def pop(self):
        self._vec = [0, *self._vec[:self.length-1]]

    def fill(self):
        if not self.filled:
            for _ in range(1, self.length):
                self.pop()
                self.lower_triang.append(self._vec)
            self.lower_triang = np.array(self.lower_triang).T
            self.filled = True

    def get(self):
        if self.filled:
            return self.lower_triang
        else:
            self.fill()
            return self.lower_triang

