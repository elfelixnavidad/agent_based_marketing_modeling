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

### Enhanced Agent Architecture

#### Advanced User Segmentation System
- **UserPersona**: Detailed user persona dataclass with comprehensive attributes
- **UserSegmentationEngine**: Behavioral targeting and segmentation logic
- **EnhancedPlayerPersona**: Integration of segmentation with Mesa agent framework
- **Behavioral Patterns**: 9 behavioral patterns for personalized targeting
- **User Segments**: 12 detailed user segments with behavioral targeting
- **Lookalike Modeling**: Similarity scoring and audience targeting

### Agent States

- `UNAWARE`: Not aware of the game
- `AWARE`: Aware but hasn't installed
- `INSTALLED`: Active player
- `PAYING_USER`: Customer who has made purchases
- `LAPSED`: Churned player

## 🎯 Key Features

### 1. Advanced User Segmentation & Behavioral Targeting
- **Detailed User Personas**: Comprehensive demographic, behavioral, and preference attributes
- **12 User Segments**: Casual, Core, Whale, Newbie, High Engagement, Low Engagement, Social Butterfly, Solitary Player, Competitive, Casual Social, Returning, Churned
- **9 Behavioral Patterns**: Price Sensitive, Brand Loyal, Trend Follower, Early Adopter, Late Adopter, Social Influencer, Achievement Seeker, Explorer, Collector
- **Behavioral Targeting**: Personalized channel preferences and message strategies
- **Lookalike Modeling**: Similarity scoring for audience expansion
- **Segmentation-based LTV**: Enhanced lifetime value predictions by segment
- **Adaptive Channel Preferences**: Dynamic preference adjustment based on behavior

### 2. Extensible Multi-Channel UA Modeling
- **Paid Social**: Facebook, Instagram campaigns
- **Video Ads**: TikTok, YouTube advertising
- **Search Ads**: Google Ads, App Store Optimization
- **Owned Channels**: Push notifications, email marketing
- **Influencer Marketing**: Paid influencer partnerships with authenticity bonuses
- **Out-of-Home (OOH)**: Billboards, transit advertising with geographic targeting
- **Programmatic Display**: Banner and native ads with advanced targeting options
- **Organic Social**: Viral content marketing with organic growth mechanics

### 2. Advanced Channel Architecture
- **Factory Pattern**: Dynamic channel creation and registration
- **Channel Types**: PAID_ACQUISITION, OWNED_MEDIA, EARNED_MEDIA, HYBRID
- **Priority System**: Configurable channel execution priorities
- **Budget Optimization**: Performance-based allocation with constraints
- **Custom Metrics**: Extensible per-channel metric tracking
- **Channel Cloning**: Easy configuration inheritance and testing

### 3. Advanced Analytics
- **Real-time KPIs**: MAU, ARPU, LTV, CAC, ROAS
- **Cohort Analysis**: Retention curves by acquisition cohort
- **Funnel Analysis**: Conversion rates across user journey
- **Channel Performance**: ROI and efficiency metrics

### 4. What-If Scenarios
- Budget allocation optimization
- Channel performance changes
- Market condition adjustments
- Retention strategy testing

### 5. Model Validation
- **Calibration**: Parameter tuning against historical data
- **Hindcasting**: Backtesting against known periods
- **Accuracy Metrics**: MAPE, RMSE validation

## 📈 Use Cases

### Strategic Questions Answered
- What's the optimal budget split between high-value vs. high-volume regions?
- At what spending level does ad saturation occur?
- How much paid UA is needed to achieve sustainable viral growth?
- What's the ROI of implementing new retention features?
- **Which user segments provide the highest LTV and retention rates?**
- **How can behavioral targeting improve campaign efficiency?**
- **What channel preferences do different user personas exhibit?**
- **How does segmentation-based targeting affect overall ROI?**

### Business Applications
- **Marketing Planning**: Test budget allocations before committing spend
- **Campaign Optimization**: Identify underperforming channels and opportunities
- **Market Analysis**: Simulate impact of competitive changes
- **Product Decisions**: Evaluate feature ROI based on retention impact
- **User Segmentation**: Identify high-value user segments and behavioral patterns
- **Personalized Marketing**: Develop targeted campaigns based on user personas
- **Audience Expansion**: Use lookalike modeling to find similar high-value users
- **Channel Strategy**: Optimize channel mix based on segment preferences

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
        'influencer': {
            'initial_budget': 25000,
            'cpi': 5.0,
            'ctr': 0.04,
            'conversion_rate': 0.15,
            'enabled': True,
            'priority': 4
        },
        'ooh': {
            'initial_budget': 15000,
            'cpi': 8.0,
            'ctr': 0.01,
            'conversion_rate': 0.05,
            'enabled': True,
            'geographic_targeting': {'cities': ['NYC', 'LA', 'SF']}
        }
    }
}

sim = MarketingSimulation(config=config)
```

### Advanced User Segmentation Configuration
```python
from src.agents.user_segmentation import UserSegmentationEngine, UserPersona
from src.agents.enhanced_player_persona import EnhancedPlayerPersona

# Create segmentation engine
engine = UserSegmentationEngine()

# Create custom user persona
persona_config = {
    "age_group": "25-34",
    "gender": "female",
    "location": "us",
    "spending_capacity": "high",
    "gaming_frequency": "heavy",
    "technical_proficiency": "advanced"
}
persona = engine.create_persona(persona_config)

# Create enhanced player persona with segmentation
class MockModel:
    def __init__(self):
        self.agents_ = {}
        self.schedule = Mock()
        self.schedule.steps = 0
        self.ua_manager = Mock()
        self.data_collector = Mock()
        from src.agents.enhanced_player_persona import EnhancedPlayerPersona
        self.agents_[EnhancedPlayerPersona] = {}

model = MockModel()
enhanced_persona = EnhancedPlayerPersona(1, model, persona_config)

# Get segmentation insights
insights = enhanced_persona.get_segmentation_insights()
print(f"User Segment: {insights['current_segment']}")
print(f"Behavioral Patterns: {insights['behavioral_patterns']}")
print(f"Channel Preferences: {insights['channel_preferences']}")

# Get enhanced LTV predictions
ltv_data = enhanced_persona.get_enhanced_ltv()
print(f"Current LTV: ${ltv_data['current_ltv']:.2f}")
print(f"Predicted LTV: ${ltv_data['predicted_ltv']:.2f}")
```

### Behavioral Targeting Configuration
```python
# Segment user based on behavior
behavioral_data = {
    "session_length": 0.7,
    "frequency": 0.8,
    "spending": 0.6,
    "social": 0.4
}

segment = engine.segment_user(persona, behavioral_data)
patterns = engine.identify_behavioral_patterns(persona)
recommendations = engine.get_targeting_recommendations(persona, segment, patterns)

print(f"Recommended Segment: {segment.value}")
print(f"Behavioral Patterns: {[p.value for p in patterns]}")
print(f"Channel Strategy: {recommendations['channel_preferences']}")
print(f"Message Strategy: {recommendations['message_strategy']}")
```

### Lookalike Modeling
```python
# Create lookalike audiences
source_persona = engine.create_persona({"spending_capacity": "high", "social_engagement": 0.8})
target_persona = engine.create_persona({"spending_capacity": "medium", "social_engagement": 0.6})

similarity_scores = engine.create_behavioral_lookalike(source_persona, target_persona)
print(f"Overall Similarity: {similarity_scores['overall_similarity']:.2f}")
print(f"Gaming Preference Similarity: {similarity_scores['gaming_preference']:.2f}")

# Calculate segment affinity
whale_affinity = engine.calculate_segment_affinity(target_persona, UserSegment.WHALE)
print(f"Whale Segment Affinity: {whale_affinity:.2f}")
```

### Dynamic Channel Creation
```python
from src.ua_channels.ua_channel import create_channel, register_channel

# Create a custom channel
custom_channel = create_channel('influencer', model, {
    'initial_budget': 30000,
    'authenticity_bonus': 1.5,
    'influencer_tiers': ['micro', 'macro', 'celebrity']
})

# Register custom channel types
register_channel('custom_partner', CustomPartnerChannel)

# Get available channels
available_channels = get_available_channels()
print(f"Available channels: {available_channels}")
```

### Channel Management
```python
# Add new channels dynamically
sim.ua_manager.add_channel('new_influencer', {
    'initial_budget': 20000,
    'cpi': 6.0,
    'priority': 5
})

# Clone channels for testing
cloned_channel = sim.ua_manager.channels['paid_social'].clone(
    'test_social',
    {'initial_budget': 5000}
)

# Get channel insights
insights = sim.ua_manager.get_channel_insights()
print(f"Top performing channel: {insights['top_performer']}")

# Optimize budget allocation
optimized = sim.ua_manager.optimize_budget_allocation('roas')
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
│   │   ├── player_persona.py           # Individual user behavior
│   │   ├── user_segmentation.py        # Advanced user segmentation system
│   │   ├── enhanced_player_persona.py  # Enhanced agent with segmentation
│   │   └── __init__.py
│   ├── environment/            # Simulation environment
│   │   ├── marketing_simulation.py
│   │   └── __init__.py
│   ├── ua_channels/           # UA channel models (Extensible)
│   │   ├── ua_channel.py      # Base class + 8 channel types
│   │   ├── ua_manager.py      # Advanced channel management
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
├── examples/                 # Example scripts
│   ├── budget_optimization.py
│   ├── cohort_analysis.py
│   └── calibration_demo.py
├── test_comprehensive.py     # Comprehensive test suite
├── test_user_segmentation.py # User segmentation tests (38 tests)
├── run_demo.py              # Quick demo script
├── requirements.txt          # Dependencies
├── claude.md                # Claude-specific documentation
├── PROJECT_OUTLINE.md       # Project architecture
├── QUICKSTART.md           # Quick start guide
└── README.md               # This file
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Core functionality tests (90 tests)
python test_comprehensive.py

# Advanced user segmentation tests (38 tests)
python test_user_segmentation.py

# Run all tests
python -m pytest test_*.py -v
```

**Test Coverage**: 128 tests covering all major functionality including:
- Agent behavior and state transitions
- Channel execution and performance tracking
- Manager operations and optimization
- Data collection and metrics
- Model calibration and validation
- Integration scenarios
- **Advanced User Segmentation**:
  - UserPersona dataclass functionality
  - UserSegmentationEngine behavioral targeting
  - EnhancedPlayerPersona integration with Mesa framework
  - Behavioral pattern identification and scoring
  - Lookalike modeling and similarity calculations
  - Segmentation-based LTV predictions
  - Channel preference adaptation
  - Targeting recommendations generation

## 🔧 Extensible Channel System

The UA channel system is designed for maximum extensibility:

### Channel Types Available:
- **PaidSocialChannel**: Facebook, Instagram, Twitter campaigns
- **VideoAdsChannel**: TikTok, YouTube, Instagram Reels
- **SearchAdsChannel**: Google Ads, App Store Optimization
- **OwnedChannel**: Push notifications, email, in-game events
- **InfluencerChannel**: Paid influencer partnerships with authenticity bonuses
- **OOHChannel**: Billboards, transit advertising with geographic targeting
- **ProgrammaticDisplayChannel**: Banner and native ads with advanced targeting
- **SocialOrganicChannel**: Viral content marketing with organic growth

### Creating Custom Channels:
```python
from src.ua_channels.ua_channel import UAChannel, ChannelType

class CustomChannel(UAChannel):
    def __init__(self, model, config=None):
        super().__init__('custom_channel', ChannelType.PAID_ACQUISITION, model, config)

    def _initialize_channel_specific_attributes(self):
        self.custom_metric = self.config.get('custom_metric', 1.0)

    def execute_campaign(self, agents):
        # Custom campaign logic
        return {'impressions': 100, 'clicks': 10, 'installs': 2, 'cost': 50}

# Register the custom channel
register_channel('custom', CustomChannel)
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
- **✅ Advanced Segmentation**: Detailed user persona modeling **(IMPLEMENTED)**
- **Real-time Data Integration**: Live campaign data feeding
- **Cross-Platform Tracking**: Unified user behavior across devices
- **Advanced A/B Testing**: Automated campaign optimization
- **Predictive Churn Modeling**: Early intervention systems
- **Real-Time Personalization**: Dynamic content adaptation
- **Market Basket Analysis**: Cross-sell and upsell optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with Mesa for agent-based modeling, Streamlit for interactive visualization, and industry best practices for mobile gaming analytics.