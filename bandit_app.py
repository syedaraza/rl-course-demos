import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------------------
# Multi-run simulation (averaged results)
# --------------------------------------
def run_bandit(k=10, steps=1000, epsilon=0.1, runs=2000):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)

    for _ in range(runs):
        true_values = np.random.normal(0, 1, k)
        q_estimates = np.zeros(k)
        action_counts = np.zeros(k)

        for t in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                action = np.argmax(q_estimates)

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
# Single-run simulation (step by step)
# --------------------------------------
def simulate_single_run(k=10, steps=50, epsilon=0.1):
    true_values = np.random.normal(0, 1, k)
    q_estimates = np.zeros(k)
    action_counts = np.zeros(k)
    history = []

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(k)
        else:
            action = np.argmax(q_estimates)

        reward = np.random.normal(true_values[action], 1)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

        history.append((t, action, reward, q_estimates.copy(), true_values))
    return history


# --------------------------------------
# Streamlit UI
# --------------------------------------
st.title("🎰 Epsilon-Greedy in K-Armed Bandit")
st.write("Explore the effect of ε (exploration rate) on learning performance.")

# Sidebar parameters
st.sidebar.header("Parameters")
k = st.sidebar.slider("Number of arms (k)", 2, 20, 10)
steps = st.sidebar.slider("Steps per run", 100, 5000, 1000, step=100)
runs = st.sidebar.slider("Number of runs (averaging)", 100, 5000, 2000, step=100)
epsilons = st.sidebar.multiselect(
    "Epsilon values to compare",
    [0.0, 0.01, 0.1, 0.2, 0.5],
    default=[0.1, 0.01, 0.0]
)

# --------------------------------------
# Averaged results
# --------------------------------------
st.subheader("📊 Averaged Results (Over Many Runs)")

results = {}
for eps in epsilons:
    avg_rewards, optimal_action_perc = run_bandit(k, steps, eps, runs)
    results[eps] = (avg_rewards, optimal_action_perc)

fig1, ax1 = plt.subplots()
for eps, (avg_rewards, _) in results.items():
    ax1.plot(avg_rewards, label=f"ε = {eps}")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Average Reward")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
for eps, (_, opt_action) in results.items():
    ax2.plot(opt_action, label=f"ε = {eps}")
ax2.set_xlabel("Steps")
ax2.set_ylabel("% Optimal Action")
ax2.legend()
st.pyplot(fig2)


# --------------------------------------
# Single run live simulation
# --------------------------------------
st.subheader("🎬 Live Single-Run Simulation")
eps_live = st.slider("Epsilon for live simulation", 0.0, 1.0, 0.1, step=0.01)
steps_live = st.slider("Steps for live simulation", 10, 200, 50, step=10)

if st.button("▶ Run Simulation"):
    history = simulate_single_run(k, steps_live, eps_live)

    placeholder = st.empty()
    for t, action, reward, q_estimates, true_values in history:
        with placeholder.container():
            st.write(f"**Step {t+1}**")
            st.write(f"Chosen action: {action}, Reward: {reward:.2f}")

            fig, ax = plt.subplots()
            indices = np.arange(k)
            bar1 = ax.bar(indices, q_estimates, alpha=0.6, label="Estimated Q(a)")
            bar2 = ax.bar(indices, true_values, alpha=0.3, label="True Value")
            bar1[action].set_color("red")  # highlight chosen action
            ax.legend()
            ax.set_xlabel("Actions")
            ax.set_ylabel("Value")
            st.pyplot(fig)
        time.sleep(0.3)  # animation delay
