# Project Specification: Agent-Based Simulation for Mobile Gaming UA Strategy

## 1. Overview

### 1.1 Project Goal
To develop a dynamic, agent-based simulation of a mobile game's user acquisition (UA) ecosystem. This model will serve as a strategic sandbox to test the potential impact of various UA strategies and budget allocation scenarios on key metrics like installs, player retention, and in-app purchase (IAP) revenue.

### 1.2 Business Problem
Traditional marketing models are backward-looking and struggle to forecast the complex, second-order effects of strategic budget shifts in the highly competitive mobile gaming market. They often fail to capture critical dynamics like organic lift from paid campaigns (k-factor virality), player churn, and LTV evolution.

### 1.3 Proposed Solution
An agent-based model (ABM) that simulates the individual behaviors of thousands of potential players ("agents"). Each agent progresses through a player lifecycle funnel, influenced by simulated UA campaigns and social interactions. This creates a "digital twin" of the game's market, enabling proactive, forward-looking strategic analysis.

### 1.4 Key Differentiators
- **Forward-Looking:** Simulates potential futures instead of only analyzing the past.
- **Systems Thinking:** Captures emergent, system-wide behavior (e.g., viral growth, market saturation) that simple regression models miss.
- **Strategic Sandbox:** Allows for risk-free testing of high-stakes "what-if" scenarios before committing UA budgets.

---
## 2. Industry Application & Validation Strategy

### 2.1 Industry Focus: Mobile Gaming
This project will focus on the free-to-play (F2P) mobile gaming industry. This sector is ideal due to the availability of public data and its clearly defined, measurable user funnels.

### 2.2 Case Study & Data Sources
The model will be calibrated and validated against historical data for a specific, publicly-traded mobile game (e.g., a title from Zynga, EA, or similar).

- **Data Sources:**
    - **Market Intelligence Services (e.g., Sensor Tower, data.ai):** For monthly estimates of downloads (installs) and IAP revenue.
    - **Company Quarterly Earnings Reports (10-Q, 10-K):** For reported marketing spend, Monthly Active Users (MAU), and official revenue figures.
    - **Industry Benchmarks:** For genre-specific retention rates (Day 1, Day 7, Day 30) and Cost Per Install (CPI).

### 2.3 Model Validation Through Hindcasting
Model "accuracy" will be established by testing its ability to reproduce historical reality. This will be a two-step process using a historical timeframe (e.g., Jan 2023 - Dec 2024).

1.  **Calibration (Jan 2023 - Dec 2023):** The model's parameters (e.g., agent `propensity_to_install`, `IAP_probability`, `churn_rate`, channel `CPI`, `organic_lift`) will be tuned. The goal is to adjust them until the simulation's outputs for 2023 (simulated installs, revenue, MAU) closely match the actual historical data for 2023.
2.  **Validation (Jan 2024 - Dec 2024):** With parameters locked from the calibration phase, the model will be fed the known marketing inputs from 2024. The simulation will run, and its outputs (predicted installs, revenue, MAU for 2024) will be compared against the actual historical data for 2024.

### 2.4 Key Accuracy Metrics
The discrepancy between the simulation's predictions and actual historical data in the validation period will be measured using:

- **Mean Absolute Percentage Error (MAPE):** To measure the accuracy of time-series predictions (e.g., simulated vs. actual monthly installs).
- **Root Mean Squared Error (RMSE):** To measure the absolute magnitude of error in key metrics (e.g., total active players).

A well-calibrated model would realistically forecast the direction and magnitude of changes in the validation period, proving its strategic value.

---
## 3. System Design

### 3.1 The Agent (Player Persona)
The fundamental unit of the simulation, defined by its states and attributes.

#### 3.1.1 States (The Player Lifecycle)
- `UNAWARE`: Default state. Not aware of the game.
- `AWARE`: Aware of the game but has not installed.
- `INSTALLED`: An active player.
- `PAYING_USER`: A subset of `INSTALLED` agents who have made an in-app purchase.
- `LAPSED`: A former player who has churned.

#### 3.1.2 Attributes (The Persona)
- `channel_preference`: Susceptibility to various UA channels.
- `install_threshold`: Score needed to move to the `INSTALLED` state.
- `IAP_probability`: Likelihood of converting to a `PAYING_USER`.
- `k_factor`: The agent's virality (word-of-mouth influence).
- `churn_propensity`: Baseline probability of becoming `LAPSED`.

### 3.2 The Environment (The Marketplace)
- **Total Addressable Market (TAM):** Total population of potential players for the game's genre.
- **Time:** Simulation proceeds in discrete time steps (e.g., 1 step = 1 day).

### 3.3 User Acquisition (UA) Levers
Global forces that influence agents based on budget allocation.

#### 3.3.1 Paid Channels (e.g., Social, Video Ads, Search)
- **Function:** Influence agents in `UNAWARE` and `AWARE` states.
- **Mechanism:** Budget is converted to `impressions` and `installs` based on a defined Cost Per Install (CPI).
- **Properties:** `CPI`, `effectiveness_multiplier`, `organic_lift_factor`.

#### 3.3.2 Owned Channels (e.g., Push Notifications, In-Game Events)
- **Function:** Influence `INSTALLED` and `PAYING_USER` agents.
- **Mechanism:** Aims to increase engagement, drive in-app purchases, and reduce churn.
- **Properties:** `reach` (limited by active player base), `retention_boost`.

---
## 4. Phased Development Plan

**Technical Stack:**
- **Language:** Python
- **Core Library:** Mesa
- **Data Handling:** Pandas
- **Visualization:** Plotly, Matplotlib
- **Dashboarding:** Streamlit or Dash
- **Package Manager** uv, brew

*(The phased plan remains structurally the same, but the goals are now specific to the gaming context, such as "Implement baseline Day 7 retention" or "Model the impact of CPI fluctuations.")*

---
## 5. Experimentation & Use Cases

### 5.1 Key Strategic Questions to Address
- **Budget Allocation:** What is the optimal UA budget split between high-cost, high-value geographic regions vs. low-cost, high-volume ones?
- **Diminishing Returns:** At what weekly spending level does a specific ad channel (e.g., TikTok) become saturated?
- **Organic Growth:** How much do we need to spend on paid UA to kickstart a self-sustaining viral loop (where k-factor > 1)?
- **Live-Ops Impact:** What is the projected 6-month ROI of developing a new in-game event that reduces churn by 5%?

---
## 6. Project Deliverables & Showcase

1.  **GitHub Repository:** A well-documented codebase with a `README.md` explaining the model, validation methodology, and setup instructions.
2.  **Technical Blog Post / Case Study:** A detailed write-up presenting the validation results (e.g., "Simulating 'Game X': A Backtest Against 2024 Market Data") and the strategic insights from simulation experiments.
3.  **Interactive Web Application:** A live dashboard where a user can adjust UA budgets and other parameters to see the forecasted impact on installs, MAU, and revenue.