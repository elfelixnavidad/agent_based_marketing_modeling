# Agent-Based Simulation for Mobile Gaming UA Strategy

An advanced agent-based modeling framework for simulating mobile gaming user acquisition strategies, providing actionable insights for marketing budget allocation and campaign optimization.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd agent_based_marketing_modeling

# Install dependencies
pip install -r requirements.txt
```

### Run Demo
```bash
# Quick simulation demo
python run_demo.py
```

### Interactive Dashboard
```bash
# Launch Streamlit dashboard
streamlit run src/visualization/dashboard.py
```

## 📊 Project Overview

This project creates a "digital twin" of a mobile game's user acquisition ecosystem, enabling strategic analysis of marketing campaigns through agent-based simulation. The model captures complex dynamics including:

- **Individual User Behavior**: Each agent has unique preferences and behaviors
- **Multi-Channel UA**: Paid social, video ads, search ads, and owned channels
- **Viral Growth**: Organic lift and k-factor modeling
- **Retention Economics**: Churn, reactivation, and LTV calculation
- **Budget Optimization**: Diminishing returns and channel effectiveness

## 🏗️ Architecture

### Core Components

1. **Agent Layer** (`src/agents/`): Individual user personas with states and behaviors
2. **Environment** (`src/environment/`): Simulation orchestration and scheduling
3. **UA Channels** (`src/ua_channels/`): Marketing channel implementations
4. **Data Collection** (`src/data_collection/`): Metrics and analytics collection
5. **Visualization** (`src/visualization/`): Interactive dashboard
6. **Calibration** (`src/calibration/`): Model validation and parameter tuning

### Agent States

- `UNAWARE`: Not aware of the game
- `AWARE`: Aware but hasn't installed
- `INSTALLED`: Active player
- `PAYING_USER`: Customer who has made purchases
- `LAPSED`: Churned player

## 🎯 Key Features

### 1. Multi-Channel UA Modeling
- **Paid Social**: Facebook, Instagram campaigns
- **Video Ads**: TikTok, YouTube advertising
- **Search Ads**: Google Ads, App Store Optimization
- **Owned Channels**: Push notifications, email marketing

### 2. Advanced Analytics
- **Real-time KPIs**: MAU, ARPU, LTV, CAC, ROAS
- **Cohort Analysis**: Retention curves by acquisition cohort
- **Funnel Analysis**: Conversion rates across user journey
- **Channel Performance**: ROI and efficiency metrics

### 3. What-If Scenarios
- Budget allocation optimization
- Channel performance changes
- Market condition adjustments
- Retention strategy testing

### 4. Model Validation
- **Calibration**: Parameter tuning against historical data
- **Hindcasting**: Backtesting against known periods
- **Accuracy Metrics**: MAPE, RMSE validation

## 📈 Use Cases

### Strategic Questions Answered
- What's the optimal budget split between high-value vs. high-volume regions?
- At what spending level does ad saturation occur?
- How much paid UA is needed to achieve sustainable viral growth?
- What's the ROI of implementing new retention features?

### Business Applications
- **Marketing Planning**: Test budget allocations before committing spend
- **Campaign Optimization**: Identify underperforming channels and opportunities
- **Market Analysis**: Simulate impact of competitive changes
- **Product Decisions**: Evaluate feature ROI based on retention impact

## 🔧 Configuration

### Basic Simulation Setup
```python
from src.environment.marketing_simulation import MarketingSimulation

# Create simulation
sim = MarketingSimulation(
    num_agents=10000,
    initial_budget=100000,
    width=100,
    height=100
)

# Run simulation
sim.run_model(steps=90)

# Get results
metrics = sim.data_collector.get_kpi_summary()
```

### Custom UA Channel Configuration
```python
config = {
    'ua_channels': {
        'paid_social': {
            'initial_budget': 50000,
            'cpi': 2.5,
            'ctr': 0.03,
            'conversion_rate': 0.08
        },
        'video_ads': {
            'initial_budget': 30000,
            'cpi': 3.0,
            'ctr': 0.025,
            'conversion_rate': 0.06
        }
    }
}

sim = MarketingSimulation(config=config)
```

## 🎮 Interactive Dashboard

Launch the Streamlit dashboard for interactive analysis:

```bash
streamlit run src/visualization/dashboard.py
```

### Dashboard Features
- **Real-time Simulation Control**: Start/stop/step simulation
- **Budget Allocation**: Interactive budget controls
- **Channel Configuration**: Adjust CPI, CTR, conversion rates
- **Time Series Analysis**: Multi-metric trend visualization
- **Funnel Analysis**: Conversion rate tracking
- **Cohort Analysis**: Retention curve visualization
- **Scenario Testing**: What-if analysis tools

## 📊 Model Calibration

### Calibration Process
```python
from src.calibration.calibration import SimulationCalibrator, create_sample_historical_data

# Create calibrator
calibrator = SimulationCalibrator()

# Load historical data
historical_data = create_sample_historical_data(days=90)

# Set target metrics
calibrator.set_target_metrics({
    'day7_retention': 0.20,
    'install_rate': 0.15,
    'arpu': 2.50
})

# Run calibration
results = calibrator.run_calibration(
    num_agents=5000,
    simulation_days=30,
    num_iterations=50
)

# Validate model
validation = calibrator.validate_model(historical_data)
```

## 📁 Project Structure

```
agent_based_marketing_modeling/
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── player_persona.py   # Individual user behavior
│   │   └── __init__.py
│   ├── environment/            # Simulation environment
│   │   ├── marketing_simulation.py
│   │   └── __init__.py
│   ├── ua_channels/           # UA channel models
│   │   ├── ua_channel.py      # Base channel class
│   │   ├── ua_manager.py      # Channel coordination
│   │   └── __init__.py
│   ├── data_collection/       # Analytics and metrics
│   │   ├── metrics_collector.py
│   │   └── __init__.py
│   ├── visualization/         # Interactive dashboard
│   │   ├── dashboard.py
│   │   └── __init__.py
│   └── calibration/          # Model validation
│       ├── calibration.py
│       └── __init__.py
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── config/                   # Configuration files
├── requirements.txt          # Dependencies
├── run_demo.py              # Quick demo script
└── README.md               # This file
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🎯 Example Scenarios

### Scenario 1: Budget Optimization
```python
# Test different budget allocations
allocations = {
    'conservative': {'paid_social': 0.4, 'video_ads': 0.3, 'search_ads': 0.3},
    'aggressive': {'paid_social': 0.6, 'video_ads': 0.3, 'search_ads': 0.1},
    'balanced': {'paid_social': 0.5, 'video_ads': 0.25, 'search_ads': 0.25}
}

for name, allocation in allocations.items():
    sim = MarketingSimulation(initial_budget=100000)
    sim.ua_manager.reallocate_budget(allocation)
    sim.run_model(steps=90)
    print(f"{name}: LTV = ${sim.data_collector.get_kpi_summary()['ltv']:.2f}")
```

### Scenario 2: Channel Performance Impact
```python
# Simulate channel performance changes
multipliers = [0.8, 1.0, 1.2, 1.5]

for mult in multipliers:
    sim = MarketingSimulation()
    sim.ua_manager.channels['paid_social'].effectiveness_multiplier = mult
    sim.run_model(steps=60)
    installs = sim.data_collector.get_kpi_summary()['total_installs']
    print(f"Performance {mult}x: {installs:,} installs")
```

## 📈 Key Metrics Explained

### User Acquisition Metrics
- **CPI (Cost Per Install)**: Average cost to acquire one user
- **CTR (Click-Through Rate)**: Percentage of impressions that result in clicks
- **Conversion Rate**: Percentage of clicks that result in installs

### Business Metrics
- **LTV (Lifetime Value)**: Average revenue per user over their lifetime
- **CAC (Customer Acquisition Cost)**: Average cost to acquire a customer
- **ROAS (Return on Ad Spend)**: Revenue generated per dollar spent
- **ARPU (Average Revenue Per User)**: Revenue per active user

### Retention Metrics
- **Day 1/7/30 Retention**: Percentage of users still active after N days
- **Churn Rate**: Percentage of users who stop using the app
- **MAU (Monthly Active Users)**: Number of unique users in the past 30 days

## 🔮 Future Enhancements

- **Machine Learning Integration**: predictive models for user behavior
- **Competitive Simulation**: Multiple games competing for users
- **Geographic Modeling**: Regional preferences and market differences
- **Advanced Segmentation**: Detailed user persona modeling
- **Real-time Data Integration**: Live campaign data feeding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with Mesa for agent-based modeling, Streamlit for interactive visualization, and industry best practices for mobile gaming analytics.