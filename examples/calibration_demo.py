#!/usr/bin/env python3
"""
Example: Model Calibration and Validation
Demonstrates how to calibrate the simulation against historical data
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.calibration.calibration import SimulationCalibrator, create_sample_historical_data


def run_calibration_demo():
    """Demonstrate model calibration process"""
    
    print("🔧 Model Calibration Demo")
    print("=" * 40)
    
    # Create calibrator
    calibrator = SimulationCalibrator()
    
    # Generate sample historical data
    print("\n📊 Generating sample historical data...")
    historical_data = create_sample_historical_data(days=90)
    print(f"Generated {len(historical_data)} days of historical data")
    
    # Set target metrics based on industry benchmarks
    target_metrics = {
        'day1_retention': 0.40,    # 40% Day 1 retention
        'day7_retention': 0.20,    # 20% Day 7 retention  
        'day30_retention': 0.10,   # 10% Day 30 retention
        'install_rate': 0.15,      # 15% install rate
        'paying_rate': 0.05,       # 5% paying user rate
        'arpu': 2.50,             # $2.50 ARPU
        'cac': 3.00               # $3.00 CAC
    }
    
    calibrator.set_target_metrics(target_metrics)
    print(f"\n🎯 Target metrics set:")
    for metric, value in target_metrics.items():
        print(f"  {metric}: {value}")
    
    # Run calibration with reduced iterations for demo
    print(f"\n🔄 Running calibration (this may take a moment)...")
    calibration_results = calibrator.run_calibration(
        num_agents=3000,        # Smaller population for faster calibration
        simulation_days=30,     # 30-day calibration period
        num_iterations=20       # Reduced iterations for demo
    )
    
    # Display calibration results
    print(f"\n🏆 Calibration Results:")
    print(f"Best Score: {calibration_results['best_score']:.4f}")
    print(f"\nBest Parameters:")
    for param, value in calibration_results['best_parameters'].items():
        print(f"  {param}: {value:.4f}")
    
    # Get metrics from best calibration run
    best_run = min(calibrator.calibration_results, key=lambda x: x['score'])
    print(f"\n📈 Achieved Metrics:")
    achieved_metrics = best_run['metrics']
    for metric, value in achieved_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # Run validation
    print(f"\n🔍 Running validation...")
    validation_data = historical_data.iloc[30:60]  # Use next 30 days for validation
    
    validation_results = calibrator.validate_model(validation_data, num_agents=3000)
    
    print(f"\n✅ Validation Results:")
    accuracy = validation_results['accuracy_metrics']
    print(f"  MAPE: {accuracy['mape']:.2%}")
    print(f"  RMSE: {accuracy['rmse']:.2f}")
    print(f"  Metrics Compared: {accuracy['num_metrics_compared']}")
    
    # Create calibration report
    report = calibrator.get_calibration_report()
    
    # Save results
    calibrator.save_results('calibration_results.json')
    print(f"\n💾 Calibration results saved to 'calibration_results.json'")
    
    # Create visualization
    create_calibration_visualization(calibrator, historical_data)
    
    return calibrator, validation_results


def create_calibration_visualization(calibrator, historical_data):
    """Create calibration visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Calibration Progress
    if calibrator.calibration_results:
        iterations = [r['iteration'] for r in calibrator.calibration_results]
        scores = [r['score'] for r in calibrator.calibration_results]
        
        ax1.plot(iterations, scores, 'b-', alpha=0.7)
        ax1.axhline(y=min(scores), color='r', linestyle='--', label=f'Best Score: {min(scores):.4f}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Calibration Score')
        ax1.set_title('Calibration Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter Distribution
    if calibrator.calibration_results:
        # Extract parameter values from all runs
        param_data = {}
        for result in calibrator.calibration_results:
            for param, value in result['parameters'].items():
                if param not in param_data:
                    param_data[param] = []
                param_data[param].append(value)
        
        # Show distribution of key parameters
        key_params = ['install_threshold', 'iap_probability', 'k_factor', 'churn_propensity']
        for i, param in enumerate(key_params):
            if param in param_data:
                ax2.hist(param_data[param], bins=10, alpha=0.7, label=param)
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Parameter Distribution')
        ax2.legend()
    
    # Plot 3: Target vs Achieved Metrics
    if calibrator.calibration_results and calibrator.target_metrics:
        best_run = min(calibrator.calibration_results, key=lambda x: x['score'])
        achieved = best_run['metrics']
        
        metrics_to_compare = list(set(calibrator.target_metrics.keys()) & set(achieved.keys()))
        
        x = range(len(metrics_to_compare))
        target_values = [calibrator.target_metrics[m] for m in metrics_to_compare]
        achieved_values = [achieved[m] for m in metrics_to_compare]
        
        ax3.bar([i - 0.2 for i in x], target_values, width=0.4, label='Target', alpha=0.7)
        ax3.bar([i + 0.2 for i in x], achieved_values, width=0.4, label='Achieved', alpha=0.7)
        
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Value')
        ax3.set_title('Target vs Achieved Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', '\n') for m in metrics_to_compare], rotation=0)
        ax3.legend()
    
    # Plot 4: Historical vs Simulated Comparison
    if historical_data is not None and len(historical_data) > 0:
        # Aggregate historical data by week
        historical_data['week'] = historical_data.index // 7
        weekly_historical = historical_data.groupby('week').agg({
            'installs': 'sum',
            'revenue': 'sum',
            'spend': 'sum'
        }).reset_index()
        
        weeks = weekly_historical['week'].values
        
        # Plot weekly installs
        ax4_twin = ax4.twinx()
        ax4.plot(weeks, weekly_historical['installs'], 'b-', label='Historical Installs', linewidth=2)
        
        # Add simulated data if available (approximation)
        if calibrator.best_parameters:
            simulated_weekly = []
            for week in weeks:
                # Simple approximation based on calibrated parameters
                base_installs = 1000 * calibrator.best_parameters.get('install_threshold', 0.5)
                simulated_weekly.append(base_installs * (1 + 0.1 * week))
            
            ax4_twin.plot(weeks, simulated_weekly, 'r--', label='Simulated Installs', linewidth=2)
        
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Historical Installs', color='b')
        ax4_twin.set_ylabel('Simulated Installs', color='r')
        ax4.set_title('Historical vs Simulated Trend')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('calibration_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Calibration visualization saved as 'calibration_analysis.png'")
    
    plt.show()


def main():
    """Run the calibration demonstration"""
    try:
        calibrator, validation_results = run_calibration_demo()
        
        print(f"\n✅ Calibration demo complete!")
        print(f"🔧 Best parameters found and validated")
        print(f"📊 Visualization and report generated")
        
        return calibrator, validation_results
        
    except Exception as e:
        print(f"❌ Error running calibration demo: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    calibrator, validation = main()