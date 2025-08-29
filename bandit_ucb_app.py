import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# --------------------------------------
# UCB multi-run simulation
# --------------------------------------
def run_ucb_bandit(k=10, steps=1000, c=2, runs=2000):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)

    for _ in range(runs):
        true_values = np.random.normal(0, 1, k)
        q_estimates = np.zeros(k)
        action_counts = np.zeros(k)

        for t in range(steps):
            if t < k:  # play each action once first
                action = t
            else:
                ucb_values = q_estimates + c * np.sqrt(np.log(t+1) / (action_counts + 1e-5))
                action = np.argmax(ucb_values)

            reward = np.random.normal(true_values[action], 1)
            action_counts[action] += 1
            q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

            rewards[t] += reward
            if action == np.argmax(true_values):
                optimal_action_counts[t] += 1

    avg_rewards = rewards / runs
    optimal_action_perc = (optimal_action_counts / runs) * 100
    return avg_rewards, optimal_action_perc


# --------------------------------------
# Single-run simulation for UCB
# --------------------------------------
def simulate_ucb_single_run(k=10, steps=50, c=2):
    true_values = np.random.normal(0, 1, k)
    q_estimates = np.zeros(k)
    action_counts = np.zeros(k)
    history = []

    for t in range(steps):
        if t < k:  # ensure each arm is tried
            action = t
        else:
            ucb_values = q_estimates + c * np.sqrt(np.log(t+1) / (action_counts + 1e-5))
            action = np.argmax(ucb_values)

        reward = np.random.normal(true_values[action], 1)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

        history.append((t, action, reward, q_estimates.copy(), true_values, action_counts.copy()))
    return history


# --------------------------------------
# Streamlit UI
# --------------------------------------
st.title("📈 Upper Confidence Bound (UCB) in K-Armed Bandit")
st.write("The **UCB method** balances exploration and exploitation by adding an exploration bonus "
         "to less-tried actions. Arms with high uncertainty are explored more often.")

# Sidebar parameters
st.sidebar.header("Parameters")
k = st.sidebar.slider("Number of arms (k)", 2, 20, 10)
steps = st.sidebar.slider("Steps per run", 100, 5000, 1000, step=100)
runs = st.sidebar.slider("Number of runs (averaging)", 100, 5000, 2000, step=100)
c_values = st.sidebar.multiselect(
    "UCB exploration parameter (c) values to compare",
    [0.5, 1, 2, 5],
    default=[2, 1]
)

# --------------------------------------
# Averaged results
# --------------------------------------
st.subheader("📊 Averaged Results (Over Many Runs)")

results = {}
for c in c_values:
    avg_rewards, optimal_action_perc = run_ucb_bandit(k, steps, c, runs)
    results[c] = (avg_rewards, optimal_action_perc)

fig1, ax1 = plt.subplots()
for c, (avg_rewards, _) in results.items():
    ax1.plot(avg_rewards, label=f"c = {c}")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Average Reward")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
for c, (_, opt_action) in results.items():
    ax2.plot(opt_action, label=f"c = {c}")
ax2.set_xlabel("Steps")
ax2.set_ylabel("% Optimal Action")
ax2.legend()
st.pyplot(fig2)


# --------------------------------------
# Single run live simulation
# --------------------------------------
st.subheader("🎬 Live Single-Run Simulation")
c_live = st.slider("Exploration parameter (c) for live simulation", 0.1, 5.0, 2.0, step=0.1)
steps_live = st.slider("Steps for live simulation", 10, 200, 50, step=10)

if st.button("▶ Run UCB Simulation"):
    history = simulate_ucb_single_run(k, steps_live, c_live)

    placeholder = st.empty()
    for t, action, reward, q_estimates, true_values, action_counts in history:
        with placeholder.container():
            st.write(f"**Step {t+1}**")
            st.write(f"Chosen action: {action}, Reward: {reward:.2f}, Count: {action_counts[action]}")

            fig, ax = plt.subplots()
            indices = np.arange(k)
            bar1 = ax.bar(indices, q_estimates, alpha=0.6, label="Estimated Q(a)")
            bar2 = ax.bar(indices, true_values, alpha=0.3, label="True Value")
            bar1[action].set_color("red")  # highlight chosen action
            ax.legend()
            ax.set_xlabel("Actions")
            ax.set_ylabel("Value")
            st.pyplot(fig)
        time.sleep(0.3)


# --------------------------------------
# Teaching notes
# --------------------------------------
st.markdown("""
### ✅ Teaching Notes
- **Main idea of UCB**:  
  - Choose actions with high estimated value **plus** an exploration bonus.  
  - Arms that have been tried less get a **larger bonus**.  
- This means UCB **systematically explores arms that might be optimal** instead of random exploration.  
- Parameter **c** controls how strong the exploration bonus is:  
  - Low c → more greedy.  
  - High c → more exploratory.  
""")
