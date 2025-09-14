#!/usr/bin/env python3
"""
Example: Budget Optimization Analysis
Demonstrates different budget allocation strategies and their impact
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.marketing_simulation import MarketingSimulation


def analyze_budget_strategies():
    """Compare different budget allocation strategies"""
    
    print("📊 Budget Optimization Analysis")
    print("=" * 50)
    
    # Define budget allocation strategies
    strategies = {
        'Equal Split': {
            'paid_social': 0.33,
            'video_ads': 0.34,
            'search_ads': 0.33
        },
        'Social Heavy': {
            'paid_social': 0.60,
            'video_ads': 0.25,
            'search_ads': 0.15
        },
        'Video Focused': {
            'paid_social': 0.25,
            'video_ads': 0.60,
            'search_ads': 0.15
        },
        'Search Optimized': {
            'paid_social': 0.20,
            'video_ads': 0.20,
            'search_ads': 0.60
        },
        'Balanced Mix': {
            'paid_social': 0.45,
            'video_ads': 0.35,
            'search_ads': 0.20
        }
    }
    
    # Simulation parameters
    total_budget = 100000
    simulation_days = 90
    num_agents = 8000
    
    results = []
    
    # Test each strategy
    for strategy_name, allocation in strategies.items():
        print(f"\n🎯 Testing: {strategy_name}")
        
        # Create simulation
        sim = MarketingSimulation(
            num_agents=num_agents,
            initial_budget=total_budget
        )
        
        # Apply budget allocation
        budget_allocation = {
            channel: total_budget * percentage
            for channel, percentage in allocation.items()
        }
        sim.ua_manager.reallocate_budget(budget_allocation)
        
        # Run simulation
        sim.run_model(steps=simulation_days)
        
        # Collect results
        kpis = sim.data_collector.get_kpi_summary()
        
        result = {
            'strategy': strategy_name,
            'total_installs': kpis['total_installs'],
            'total_revenue': kpis['total_revenue'],
            'ltv': kpis['ltv'],
            'cac': kpis['cac'],
            'ltv_cac_ratio': kpis['ltv_cac_ratio'],
            'roas': kpis['roas'],
            'day7_retention': kpis['day7_retention'],
            'mau': kpis['current_mau']
        }
        
        results.append(result)
        
        # Print key metrics
        print(f"  Installs: {result['total_installs']:,}")
        print(f"  Revenue: ${result['total_revenue']:,.0f}")
        print(f"  LTV: ${result['ltv']:.2f}")
        print(f"  CAC: ${result['cac']:.2f}")
        print(f"  LTV:CAC: {result['ltv_cac_ratio']:.2f}")
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    print(f"\n🏆 Strategy Rankings:")
    
    # Rank by different metrics
    rankings = {
        'By Installs': results_df.nlargest(5, 'total_installs')[['strategy', 'total_installs']],
        'By Revenue': results_df.nlargest(5, 'total_revenue')[['strategy', 'total_revenue']],
        'By LTV:CAC': results_df.nlargest(5, 'ltv_cac_ratio')[['strategy', 'ltv_cac_ratio']],
        'By ROAS': results_df.nlargest(5, 'roas')[['strategy', 'roas']]
    }
    
    for ranking_name, ranking_df in rankings.items():
        print(f"\n  {ranking_name}:")
        for _, row in ranking_df.iterrows():
            value_col = ranking_df.columns[1]
            print(f"    {row['strategy']}: {row[value_col]:,.0f}" if value_col in ['total_installs', 'total_revenue'] 
                  else f"    {row['strategy']}: {row[value_col]:.2f}")
    
    # Identify best overall strategy
    # Composite score based on multiple metrics
    results_df['composite_score'] = (
        results_df['total_installs'] / results_df['total_installs'].max() * 0.25 +
        results_df['total_revenue'] / results_df['total_revenue'].max() * 0.25 +
        results_df['ltv_cac_ratio'] / results_df['ltv_cac_ratio'].max() * 0.25 +
        results_df['roas'] / results_df['roas'].max() * 0.25
    )
    
    best_strategy = results_df.loc[results_df['composite_score'].idxmax()]
    
    print(f"\n🥇 Best Overall Strategy: {best_strategy['strategy']}")
    print(f"   Composite Score: {best_strategy['composite_score']:.3f}")
    
    # Visualize results
    create_strategy_visualization(results_df, strategies)
    
    return results_df


def create_strategy_visualization(results_df, strategies):
    """Create visualization of budget strategy results"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Strategy names for x-axis
    strategy_names = results_df['strategy'].values
    
    # Plot 1: Installs and Revenue
    x = range(len(strategy_names))
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar([i - 0.2 for i in x], results_df['total_installs'], 
                    width=0.4, label='Installs', alpha=0.7, color='blue')
    bars2 = ax1_twin.bar([i + 0.2 for i in x], results_df['total_revenue'], 
                         width=0.4, label='Revenue ($)', alpha=0.7, color='green')
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Installs', color='blue')
    ax1_twin.set_ylabel('Revenue ($)', color='green')
    ax1.set_title('Acquisition & Revenue by Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(' ', '\n') for name in strategy_names], rotation=0)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: LTV vs CAC
    ax2.scatter(results_df['cac'], results_df['ltv'], s=100, alpha=0.7)
    for i, strategy in enumerate(strategy_names):
        ax2.annotate(strategy, (results_df['cac'].iloc[i], results_df['ltv'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('CAC ($)')
    ax2.set_ylabel('LTV ($)')
    ax2.set_title('LTV vs CAC by Strategy')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency Metrics
    efficiency_metrics = ['ltv_cac_ratio', 'roas']
    x_eff = range(len(strategy_names))
    width = 0.35
    
    bars3 = ax3.bar([i - width/2 for i in x_eff], results_df['ltv_cac_ratio'], 
                    width, label='LTV:CAC Ratio', alpha=0.7)
    bars4 = ax3.bar([i + width/2 for i in x_eff], results_df['roas'], 
                    width, label='ROAS', alpha=0.7)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Ratio')
    ax3.set_title('Efficiency Metrics by Strategy')
    ax3.set_xticks(x_eff)
    ax3.set_xticklabels([name.replace(' ', '\n') for name in strategy_names], rotation=0)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Budget Allocation Heatmap
    allocation_data = []
    for strategy in strategy_names:
        allocation = strategies[strategy]
        row = [allocation.get('paid_social', 0), 
               allocation.get('video_ads', 0), 
               allocation.get('search_ads', 0)]
        allocation_data.append(row)
    
    im = ax4.imshow(allocation_data, cmap='Blues', aspect='auto')
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(len(strategy_names)))
    ax4.set_xticklabels(['Paid Social', 'Video Ads', 'Search Ads'])
    ax4.set_yticklabels([name.replace(' ', '\n') for name in strategy_names])
    ax4.set_title('Budget Allocation by Strategy')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Budget Share')
    
    # Add percentage labels
    for i in range(len(strategy_names)):
        for j in range(3):
            text = ax4.text(j, i, f'{allocation_data[i][j]:.0%}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('budget_strategy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'budget_strategy_analysis.png'")
    
    # Show plot
    plt.show()


def main():
    """Run the budget optimization analysis"""
    try:
        results = analyze_budget_strategies()
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Detailed results saved in results dataframe")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return None


if __name__ == "__main__":
    results = main()