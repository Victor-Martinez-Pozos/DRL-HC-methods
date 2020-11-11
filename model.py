import numpy as np


class FCNetwork():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        
    def forward(self, state):
        x = np.dot(state, self.w)
        x = np.exp(x)/sum(np.exp(x))
        return x
    
    def save(self, name):
        with open(f'{name}.npy', 'wb') as f:
            np.save(f, self.w)

    def load(self, name):
        with open(f'{name}', 'wb') as f:
            np.load(f)

