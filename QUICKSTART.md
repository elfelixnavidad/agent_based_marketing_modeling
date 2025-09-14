# 🚀 Quick Start Guide

## Installation & Setup

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python run_demo.py
   ```

## Running the Simulation

### Option 1: Quick Demo
```bash
python run_demo.py
```

### Option 2: Interactive Dashboard
```bash
# Method 1: Direct launch
python -m streamlit run src/visualization/dashboard.py

# Method 2: Use launch script
python launch_dashboard.py
```

### Option 3: Comprehensive Testing
```bash
# Run full test suite (90 tests, 100% coverage)
python test_comprehensive.py
```

### Option 4: Example Analyses
```bash
# Budget optimization analysis
python examples/budget_optimization.py

# Cohort analysis and retention modeling
python examples/cohort_analysis.py

# Model calibration and validation
python examples/calibration_demo.py
```

## Dashboard Features

### 🎛️ **Simulation Controls**
- Initialize simulation with custom parameters
- Adjust budget allocation across UA channels
- Configure channel performance settings
- Run simulation step-by-step or continuously

### 📊 **Real-time Analytics**
- **KPI Dashboard**: MAU, ARPU, LTV, CAC, ROAS
- **Time Series**: Acquisition, revenue, retention trends
- **Funnel Analysis**: Conversion rates across user journey
- **Cohort Analysis**: Retention curves by acquisition cohort

### 📢 **Channel Performance**
- **8 Channel Types**: Paid Social, Video Ads, Search, Influencer, OOH, Programmatic, Organic, Owned
- **Advanced Metrics**: CPI, CTR, conversion rate, authenticity bonuses, organic lift
- **Budget Efficiency**: ROI tracking and performance optimization
- **Channel Comparison**: Side-by-side performance visualization
- **Custom Metrics**: Extensible per-channel performance tracking

### 🔮 **What-If Scenarios**
- Budget allocation optimization
- Channel performance sensitivity analysis
- Market condition simulation
- Retention strategy testing

## Key Concepts

### **Agent States**
- `UNAWARE` → `AWARE` → `INSTALLED` → `PAYING_USER` → `LAPSED`

### **UA Channels**
- **Paid Social**: Facebook, Instagram campaigns
- **Video Ads**: TikTok, YouTube advertising
- **Search Ads**: Google Ads, App Store Optimization
- **Owned Channels**: Push notifications, email marketing
- **Influencer Marketing**: Paid partnerships with authenticity bonuses
- **Out-of-Home (OOH)**: Billboards and transit advertising
- **Programmatic Display**: Banner and native ads with advanced targeting
- **Organic Social**: Viral content marketing with organic growth

### **New Extensible Features**
- **Dynamic Channel Creation**: Add custom channels at runtime
- **Channel Factory Pattern**: Easy channel instantiation and registration
- **Performance Optimization**: Automated budget allocation based on ROI
- **Channel Cloning**: Configuration inheritance for A/B testing
- **Priority System**: Configurable channel execution order
- **Custom Metrics**: Extensible per-channel performance tracking

### **Key Metrics**
- **LTV**: Lifetime Value per user
- **CAC**: Customer Acquisition Cost
- **ROAS**: Return on Ad Spend
- **ARPU**: Average Revenue Per User
- **Retention**: Day 1/7/30 retention rates

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the project root directory:
```bash
cd /path/to/agent_based_marketing_modeling
python run_demo.py
```

### Missing Dependencies
Install required packages:
```bash
uv pip install -r requirements.txt
```

### Dashboard Not Loading
Ensure Streamlit is installed and try:
```bash
streamlit run src/visualization/dashboard.py --server.port=8501
```

## Next Steps

1. **Run the demo** (`python run_demo.py`) to see basic functionality
2. **Launch the dashboard** for interactive exploration and what-if analysis
3. **Run comprehensive tests** (`python test_comprehensive.py`) to verify all features
4. **Try the examples** for advanced analysis and calibration
5. **Create custom channels** using the extensible factory pattern
6. **Optimize budget allocation** using performance-based algorithms
7. **Calibrate the model** using your historical data for accurate predictions

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the comprehensive README.md
- Examine the example scripts for usage patterns