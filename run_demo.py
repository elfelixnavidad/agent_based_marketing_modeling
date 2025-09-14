#!/usr/bin/env python3
"""
Quick start script for the Agent-Based Marketing Simulation
"""

import sys
import os

# Add current directory to path to access src
sys.path.append(os.path.dirname(__file__))

from src.environment.marketing_simulation import MarketingSimulation
from src.data_collection.metrics_collector import MetricsCollector
from src.ua_channels.ua_manager import UAManager
from src.agents.player_persona import PlayerState


def main():
    """Run a quick simulation demonstration"""
    print("🚀 Agent-Based Marketing Simulation Demo")
    print("=" * 50)
    
    # Create simulation
    print("Initializing simulation...")
    sim = MarketingSimulation(
        num_agents=5000,
        initial_budget=50000,
        width=50,
        height=50
    )
    
    print(f"Simulation created with {sim.num_agents} agents")
    print(f"Initial budget: ${sim.initial_budget:,}")
    
    # Display initial state
    print("\n📊 Initial State:")
    initial_stats = sim.get_summary_stats()
    for key, value in initial_stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # Run simulation
    print(f"\n🎯 Running simulation for 30 days...")
    sim.run_model(steps=30)
    
    # Display results
    print("\n📈 Final Results:")
    final_stats = sim.get_summary_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # Calculate basic metrics from simulation state
    total_installs = final_stats['Installed'] + final_stats['Paying_Users']
    total_revenue = final_stats['Total_Revenue']
    total_spend = sim.initial_budget - final_stats['Remaining_Budget']
    cac = total_spend / max(total_installs, 1)
    
    print(f"\n🎯 Key Performance Indicators:")
    print(f"  Total Revenue: ${total_revenue:,.2f}")
    print(f"  Total Installs: {total_installs:,}")
    print(f"  Total Spend: ${total_spend:,.0f}")
    print(f"  CAC: ${cac:.2f}")
    print(f"  Remaining Budget: ${final_stats['Remaining_Budget']:,.0f}")
    
    # UA Channel Performance
    print(f"\n📢 UA Channel Performance:")
    channel_performance = sim.ua_manager.get_performance_metrics()
    for channel, metrics in channel_performance.items():
        if metrics['installs'] > 0:
            print(f"  {channel.replace('_', ' ').title()}:")
            print(f"    Installs: {metrics['installs']:,}")
            print(f"    Spend: ${metrics['spend']:,.0f}")
            print(f"    CPI: ${metrics['cpi']:.2f}")
            print(f"    ROAS: {metrics['roas']:.2f}")
    
    print(f"\n✅ Simulation complete!")
    print(f"💡 To explore the interactive dashboard, run:")
    print(f"   python -m streamlit run src/visualization/dashboard.py")


if __name__ == "__main__":
    main()