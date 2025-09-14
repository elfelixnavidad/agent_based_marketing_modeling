#!/usr/bin/env python3
"""
Example: Cohort Analysis and Retention Modeling
Demonstrates advanced user retention analysis
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.marketing_simulation import MarketingSimulation


def run_cohort_analysis():
    """Run comprehensive cohort analysis"""
    
    print("👥 Cohort Analysis Demo")
    print("=" * 40)
    
    # Create simulation with specific retention parameters
    print("\n🎮 Initializing simulation...")
    sim = MarketingSimulation(
        num_agents=10000,
        initial_budget=80000
    )
    
    # Adjust retention parameters for realistic patterns
    print("⚙️  Configuring retention parameters...")
    
    # Set different churn patterns for different user segments
    for agent in sim.schedule.agents[:len(sim.schedule.agents)//3]:
        # High-value segment (lower churn)
        agent.churn_propensity *= 0.7
        agent.iap_probability *= 1.5
    
    for agent in sim.schedule.agents[len(sim.schedule.agents)//3:2*len(sim.schedule.agents)//3]:
        # Mid-value segment (normal churn)
        pass
    
    for agent in sim.schedule.agents[2*len(sim.schedule.agents)//3:]:
        # Low-value segment (higher churn)
        agent.churn_propensity *= 1.3
        agent.iap_probability *= 0.7
    
    # Run simulation for extended period to capture retention
    print(f"\n🔄 Running simulation for 120 days...")
    sim.run_model(steps=120)
    
    # Analyze cohort data
    print(f"\n📊 Analyzing cohort data...")
    cohort_data = sim.data_collector.get_cohort_analysis()
    
    if not cohort_data:
        print("❌ No cohort data available")
        return None
    
    # Calculate key retention metrics
    retention_summary = calculate_retention_metrics(cohort_data)
    
    # Display results
    print(f"\n📈 Retention Summary:")
    for metric, value in retention_summary.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value}")
    
    # Analyze LTV by cohort
    ltv_analysis = analyze_ltv_by_cohort(sim)
    
    print(f"\n💰 LTV Analysis by Cohort:")
    for cohort_week, ltv_data in ltv_analysis.items():
        print(f"  Week {cohort_week}: LTV = ${ltv_data['ltv']:.2f}, "
              f"Users = {ltv_data['user_count']}, "
              f"Paying % = {ltv_data['paying_percentage']:.1%}")
    
    # Create visualizations
    create_cohort_visualizations(cohort_data, ltv_analysis, retention_summary)
    
    # Generate retention insights
    generate_retention_insights(retention_summary, ltv_analysis)
    
    return {
        'cohort_data': cohort_data,
        'retention_summary': retention_summary,
        'ltv_analysis': ltv_analysis,
        'simulation': sim
    }


def calculate_retention_metrics(cohort_data):
    """Calculate comprehensive retention metrics"""
    
    metrics = {
        'day1_retention': [],
        'day7_retention': [],
        'day30_retention': [],
        'day90_retention': []
    }
    
    for cohort_week, retention_points in cohort_data.items():
        for point in retention_points:
            day = point['day']
            rate = point['retention_rate']
            
            if day == 1:
                metrics['day1_retention'].append(rate)
            elif day == 7:
                metrics['day7_retention'].append(rate)
            elif day == 30:
                metrics['day30_retention'].append(rate)
            elif day == 90:
                metrics['day90_retention'].append(rate)
    
    # Calculate averages
    summary = {}
    for metric, values in metrics.items():
        if values:
            summary[metric] = np.mean(values)
        else:
            summary[metric] = 0.0
    
    # Calculate additional metrics
    summary['total_cohorts'] = len(cohort_data)
    summary['avg_cohort_size'] = np.mean([
        len(retention_points) for retention_points in cohort_data.values()
    ]) if cohort_data else 0
    
    return summary


def analyze_ltv_by_cohort(sim):
    """Analyze lifetime value by acquisition cohort"""
    
    # Group agents by install week (cohort)
    cohort_analysis = {}
    
    for agent in sim.schedule.agents:
        if agent.install_time is not None:
            cohort_week = agent.install_time // 7
            
            if cohort_week not in cohort_analysis:
                cohort_analysis[cohort_week] = {
                    'users': [],
                    'total_revenue': 0.0,
                    'total_users': 0,
                    'paying_users': 0
                }
            
            cohort_analysis[cohort_week]['users'].append(agent)
            cohort_analysis[cohort_week]['total_revenue'] += agent.total_spend
            cohort_analysis[cohort_week]['total_users'] += 1
            
            if agent.total_spend > 0:
                cohort_analysis[cohort_week]['paying_users'] += 1
    
    # Calculate LTV and other metrics for each cohort
    ltv_summary = {}
    
    for cohort_week, data in cohort_analysis.items():
        if data['total_users'] > 0:
            ltv = data['total_revenue'] / data['total_users']
            paying_percentage = data['paying_users'] / data['total_users']
            
            ltv_summary[cohort_week] = {
                'ltv': ltv,
                'user_count': data['total_users'],
                'paying_users': data['paying_users'],
                'paying_percentage': paying_percentage,
                'total_revenue': data['total_revenue']
            }
    
    return ltv_summary


def create_cohort_visualizations(cohort_data, ltv_analysis, retention_summary):
    """Create comprehensive cohort visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Retention Curves by Cohort
    ax1 = plt.subplot(2, 3, 1)
    
    for cohort_week, retention_points in cohort_data.items():
        if len(retention_points) > 1:  # Only plot cohorts with multiple data points
            days = [point['day'] for point in retention_points]
            rates = [point['retention_rate'] for point in retention_points]
            ax1.plot(days, rates, marker='o', label=f'Week {cohort_week}', alpha=0.7)
    
    ax1.set_xlabel('Days Since Install')
    ax1.set_ylabel('Retention Rate')
    ax1.set_title('Retention Curves by Cohort')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Cohort Heatmap
    ax2 = plt.subplot(2, 3, 2)
    
    # Create heatmap data
    max_days = 90
    all_cohorts = sorted(cohort_data.keys())
    
    heatmap_data = []
    for cohort_week in all_cohorts:
        retention_rates = []
        for day in range(0, max_days + 1, 7):
            rate = 0.0
            for point in cohort_data[cohort_week]:
                if point['day'] == day:
                    rate = point['retention_rate']
                    break
            retention_rates.append(rate)
        heatmap_data.append(retention_rates)
    
    if heatmap_data:
        im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Weeks Since Install')
        ax2.set_ylabel('Acquisition Week')
        ax2.set_title('Cohort Retention Heatmap')
        
        # Set ticks
        week_ticks = list(range(0, max_days // 7 + 1, 2))
        ax2.set_xticks(week_ticks)
        ax2.set_xticklabels([str(w) for w in week_ticks])
        ax2.set_yticks(range(len(all_cohorts)))
        ax2.set_yticklabels([f'W{w}' for w in all_cohorts])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Retention Rate')
    
    # Plot 3: Benchmark Retention Comparison
    ax3 = plt.subplot(2, 3, 3)
    
    # Industry benchmarks (approximate)
    benchmarks = {
        'Day 1': 0.40,
        'Day 7': 0.20,
        'Day 30': 0.10,
        'Day 90': 0.05
    }
    
    metrics = ['day1_retention', 'day7_retention', 'day30_retention', 'day90_retention']
    labels = ['Day 1', 'Day 7', 'Day 30', 'Day 90']
    
    our_rates = [retention_summary.get(metric, 0) for metric in metrics]
    benchmark_rates = [benchmarks[label] for label in labels]
    
    x = range(len(labels))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], our_rates, width, label='Our Game', alpha=0.7)
    ax3.bar([i + width/2 for i in x], benchmark_rates, width, label='Industry Avg', alpha=0.7)
    
    ax3.set_xlabel('Retention Period')
    ax3.set_ylabel('Retention Rate')
    ax3.set_title('Retention vs Industry Benchmarks')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.set_ylim(0, max(max(our_rates), max(benchmark_rates)) * 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: LTV by Cohort
    ax4 = plt.subplot(2, 3, 4)
    
    if ltv_analysis:
        cohorts = sorted(ltv_analysis.keys())
        ltvs = [ltv_analysis[cohort]['ltv'] for cohort in cohorts]
        user_counts = [ltv_analysis[cohort]['user_count'] for cohort in cohorts]
        
        # Normalize user counts for bubble size
        max_users = max(user_counts) if user_counts else 1
        bubble_sizes = [count / max_users * 300 + 50 for count in user_counts]
        
        scatter = ax4.scatter(cohorts, ltvs, s=bubble_sizes, c=user_counts, 
                              cmap='viridis', alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('Acquisition Week')
        ax4.set_ylabel('LTV ($)')
        ax4.set_title('LTV by Acquisition Cohort')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for user count
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Cohort Size')
        
        # Add value labels on bubbles
        for i, (cohort, ltv) in enumerate(zip(cohorts, ltvs)):
            ax4.annotate(f'${ltv:.1f}', (cohort, ltv), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Plot 5: Paying User Percentage by Cohort
    ax5 = plt.subplot(2, 3, 5)
    
    if ltv_analysis:
        cohorts = sorted(ltv_analysis.keys())
        paying_percentages = [ltv_analysis[cohort]['paying_percentage'] for cohort in cohorts]
        
        bars = ax5.bar(cohorts, paying_percentages, alpha=0.7, color='orange')
        ax5.set_xlabel('Acquisition Week')
        ax5.set_ylabel('Paying User Percentage')
        ax5.set_title('Monetization by Cohort')
        ax5.set_ylim(0, max(paying_percentages) * 1.1 if paying_percentages else 0.1)
        ax5.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, percentage in zip(bars, paying_percentages):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{percentage:.1%}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Retention Decay Curve
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate average retention across all cohorts
    all_retention_data = {}
    for cohort_data_points in cohort_data.values():
        for point in cohort_data_points:
            day = point['day']
            rate = point['retention_rate']
            if day not in all_retention_data:
                all_retention_data[day] = []
            all_retention_data[day].append(rate)
    
    # Calculate averages
    avg_retention = {}
    for day, rates in all_retention_data.items():
        avg_retention[day] = np.mean(rates)
    
    # Sort by day
    sorted_days = sorted(avg_retention.keys())
    sorted_rates = [avg_retention[day] for day in sorted_days]
    
    ax6.plot(sorted_days, sorted_rates, 'b-', linewidth=2, marker='o', markersize=4)
    ax6.set_xlabel('Days Since Install')
    ax6.set_ylabel('Average Retention Rate')
    ax6.set_title('Average Retention Decay Curve')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    # Fit exponential decay curve
    if len(sorted_days) > 3:
        # Simple exponential fit: retention = a * exp(-b * day)
        try:
            log_rates = np.log(sorted_rates)
            coeffs = np.polyfit(sorted_days, log_rates, 1)
            a = np.exp(coeffs[1])
            b = -coeffs[0]
            
            # Generate fitted curve
            fitted_days = np.linspace(0, max(sorted_days), 100)
            fitted_rates = a * np.exp(-b * fitted_days)
            
            ax6.plot(fitted_days, fitted_rates, 'r--', 
                    label=f'Fit: {a:.2f} * exp(-{b:.3f} * day)', alpha=0.8)
            ax6.legend()
        except:
            pass
    
    plt.tight_layout()
    plt.savefig('cohort_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Cohort analysis visualization saved as 'cohort_analysis.png'")
    
    plt.show()


def generate_retention_insights(retention_summary, ltv_analysis):
    """Generate actionable retention insights"""
    
    print(f"\n💡 Retention Insights:")
    print("=" * 30)
    
    # Overall retention health
    day7_retention = retention_summary.get('day7_retention', 0)
    day30_retention = retention_summary.get('day30_retention', 0)
    
    if day7_retention >= 0.25:
        print(f"✅ Strong Day 7 retention ({day7_retention:.1%}) - above industry average")
    elif day7_retention >= 0.15:
        print(f"⚠️  Moderate Day 7 retention ({day7_retention:.1%}) - room for improvement")
    else:
        print(f"❌ Low Day 7 retention ({day7_retention:.1%}) - urgent attention needed")
    
    # Retention curve analysis
    retention_decay = day30_retention / day7_retention if day7_retention > 0 else 0
    
    if retention_decay >= 0.5:
        print(f"✅ Healthy retention decay (Day 30/Day 7 = {retention_decay:.1%})")
    elif retention_decay >= 0.3:
        print(f"⚠️  Moderate retention decay (Day 30/Day 7 = {retention_decay:.1%})")
    else:
        print(f"❌ Poor retention decay (Day 30/Day 7 = {retention_decay:.1%})")
    
    # LTV trends
    if ltv_analysis:
        ltvs = [data['ltv'] for data in ltv_analysis.values()]
        if len(ltvs) > 1:
            ltv_trend = ltvs[-1] - ltvs[0]  # Change from first to last cohort
            
            if ltv_trend > 0:
                print(f"✅ Improving LTV trend (+${ltv_trend:.2f} from first to last cohort)")
            elif ltv_trend < -0.5:
                print(f"❌ Declining LTV trend (${ltv_trend:.2f} from first to last cohort)")
            else:
                print(f"➡️  Stable LTV trend (${ltv_trend:.2f} change)")
    
    # Monetization insights
    if ltv_analysis:
        paying_percentages = [data['paying_percentage'] for data in ltv_analysis.values()]
        avg_paying_rate = np.mean(paying_percentages)
        
        if avg_paying_rate >= 0.08:
            print(f"✅ Strong monetization ({avg_paying_rate:.1%} paying users)")
        elif avg_paying_rate >= 0.04:
            print(f"⚠️  Moderate monetization ({avg_paying_rate:.1%} paying users)")
        else:
            print(f"❌ Low monetization ({avg_paying_rate:.1%} paying users)")
    
    # Actionable recommendations
    print(f"\n🎯 Recommendations:")
    
    if day7_retention < 0.20:
        print(f"  • Implement Day 7 retention campaigns (push notifications, special offers)")
    
    if retention_decay < 0.4:
        print(f"  • Improve mid-game content and engagement features")
    
    if ltv_analysis and avg_paying_rate < 0.05:
        print(f"  • Optimize IAP pricing and placement")
        print(f"  • Add subscription or battle pass options")
    
    print(f"  • Use cohort analysis to identify high-value user segments")
    print(f"  • A/B test different onboarding flows for new users")


def main():
    """Run the cohort analysis demonstration"""
    try:
        results = run_cohort_analysis()
        
        print(f"\n✅ Cohort analysis complete!")
        print(f"📊 Comprehensive retention insights generated")
        print(f"📈 Visualizations saved for analysis")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running cohort analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()