import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Bandit class
class Bandit:
    def __init__(self, p):
        # Probability of winning for the bandit
        self.p = p
        self.q_estimate = 0  # Estimated probability
        self.n_pulls = 0      # Number of times this bandit has been pulled
    
    def pull(self):
        # Simulate pulling the arm of the bandit
        return np.random.rand() < self.p

    def update(self, reward):
        # Update the estimate of q using the incremental formula
        self.n_pulls += 1
        self.q_estimate += (reward - self.q_estimate) / self.n_pulls

# Define the multi-armed bandit experiment
def run_experiment(bandit_probs, n_trials, eps):
    bandits = [Bandit(p) for p in bandit_probs]
    rewards = np.zeros(n_trials)

    for i in range(n_trials):
        # Epsilon-greedy action selection
        if np.random.rand() < eps:
            # Explore: choose a random bandit
            j = np.random.randint(len(bandits))
        else:
            # Exploit: choose the bandit with the best estimated probability
            j = np.argmax([b.q_estimate for b in bandits])
        
        # Pull the selected bandit's arm
        reward = bandits[j].pull()
        rewards[i] = reward
        
        # Update the estimated probability of the chosen bandit
        bandits[j].update(reward)
    
    return rewards, bandits

# Streamlit app starts here
st.title('Multi-Armed Bandit Simulation')

# Inputs from the user
n_arms = st.slider('Number of Bandit Arms', 2, 10, 3)
n_trials = st.slider('Number of Trials', 100, 500, 100)
eps = st.slider('Exploration Rate (ε)', 0.01, 1.0, 0.1, step=0.01)

# Initialize random probabilities for each bandit
bandit_probs = np.random.rand(n_arms)

# Display the actual probabilities
st.write(f"Actual probabilities of each bandit arm: {bandit_probs}")

# Run the experiment
if st.button('Run Experiment'):
    rewards, bandits = run_experiment(bandit_probs, n_trials, eps)
    
    # Calculate cumulative reward
    cumulative_rewards = np.cumsum(rewards)
    avg_rewards = cumulative_rewards / (np.arange(n_trials) + 1)
    
    # Plot the average reward over time
    fig, ax = plt.subplots()
    ax.plot(avg_rewards)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward Over Time')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    # Display the estimated probabilities
    st.write("Estimated probabilities for each bandit:")
    for i, bandit in enumerate(bandits):
        st.write(f"Bandit {i+1}: {bandit.q_estimate:.2f} (True: {bandit.p:.2f})")
