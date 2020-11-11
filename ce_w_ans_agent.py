import numpy as np
from collections import deque

class HillClimbing():

    def train(self, env, model, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2, pop_size=50, elite_frac=0.2):
        """Implementation of hill climbing with cross-entropy and adaptive noise scaling.
            
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
            noise_scale (float): standard deviation of additive noise
            pop_size (int): size of population at each iteration
            elite_frac (float): percentage of top performers to use in update
        """
        scores_deque = deque(maxlen=100)
        scores = []
        best_R = -np.Inf
        best_w = model.w
        n_elite=int(pop_size*elite_frac)
    
        for i_episode in range(1, n_episodes+1):
            rewards = []
            state = env.reset()
            weights_pop = [best_w + (noise_scale * np.random.rand(*model.w.shape)) for i in range(pop_size)]
            rewards = np.array([self._evaluate(env, model, weights, gamma, max_t) for weights in weights_pop])        
    
            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            cross_entropy_weight = np.array(elite_weights).mean(axis=0)
            R = self._evaluate(env, model, cross_entropy_weight, gamma, max_t)
            
            scores_deque.append(R)
            scores.append(R)
    
            if R >= best_R: # found better weights
                best_R = R
                best_w = cross_entropy_weight
                noise_scale = max(1e-3, noise_scale / 2)
            else: # did not find better weights
                noise_scale = min(2, noise_scale * 2)
    
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                model.w = best_w
                model.save("chpt_195")
                break
            
        return scores
          

    def _evaluate(self, env, model, weights, gamma=1.0, max_t=5000):
        model.w=weights
        episode_return = 0.0
        state = env.reset()
        for t in range(max_t):
            action = np.random.choice(2, p=model.forward(state)) 
            state, reward, done, _ = env.step(action)
            episode_return += reward * gamma**t
            if done:
                break 
        return episode_return



"""
        for t in range(max_t):
            action = np.random.choice(2, p=model.forward(state)) 
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

"""

    