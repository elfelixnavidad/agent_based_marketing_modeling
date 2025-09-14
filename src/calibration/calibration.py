import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.environment.marketing_simulation import MarketingSimulation


class SimulationCalibrator:
    """Handles calibration and validation of the simulation model"""
    
    def __init__(self):
        self.best_parameters = {}
        self.calibration_results = []
        self.validation_results = {}
        self.historical_data = None
        self.target_metrics = {
            'day1_retention': 0.40,
            'day7_retention': 0.20,
            'day30_retention': 0.10,
            'install_rate': 0.15,
            'paying_rate': 0.05,
            'arpu': 2.50,
            'cac': 3.00
        }
    
    def load_historical_data(self, filepath: str) -> pd.DataFrame:
        """Load historical data for calibration"""
        try:
            if filepath.endswith('.csv'):
                self.historical_data = pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.historical_data = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            print(f"Loaded historical data with {len(self.historical_data)} records")
            return self.historical_data
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None
    
    def set_target_metrics(self, targets: Dict[str, float]):
        """Set target metrics for calibration"""
        self.target_metrics.update(targets)
    
    def generate_parameter_space(self) -> Dict[str, List]:
        """Generate parameter space for calibration"""
        return {
            'agent_attributes': {
                'install_threshold': np.linspace(0.2, 0.8, 7),
                'iap_probability': np.linspace(0.01, 0.20, 8),
                'k_factor': np.linspace(0.05, 0.50, 6),
                'churn_propensity': np.linspace(0.01, 0.08, 8),
            },
            'ua_channels': {
                'paid_social_cpi': np.linspace(1.5, 4.0, 6),
                'paid_social_ctr': np.linspace(0.02, 0.05, 7),
                'paid_social_conversion': np.linspace(0.05, 0.12, 8),
                
                'video_ads_cpi': np.linspace(2.0, 5.0, 7),
                'video_ads_ctr': np.linspace(0.015, 0.035, 5),
                'video_ads_conversion': np.linspace(0.04, 0.10, 7),
                
                'search_ads_cpi': np.linspace(3.0, 6.0, 7),
                'search_ads_ctr': np.linspace(0.03, 0.08, 6),
                'search_ads_conversion': np.linspace(0.08, 0.15, 8),
            }
        }
    
    def run_calibration(self, num_agents: int = 5000, simulation_days: int = 30, 
                        num_iterations: int = 50) -> Dict[str, Any]:
        """Run calibration process"""
        print("🔧 Starting calibration process...")
        
        parameter_space = self.generate_parameter_space()
        best_score = float('inf')
        best_params = {}
        
        for iteration in range(num_iterations):
            # Generate random parameter combination
            params = self._sample_parameters(parameter_space)
            
            # Run simulation with these parameters
            sim = self._create_simulation_with_params(params, num_agents)
            sim.run_model(steps=simulation_days)
            
            # Calculate calibration score
            score = self._calculate_calibration_score(sim)
            
            # Store results
            result = {
                'iteration': iteration,
                'parameters': params,
                'score': score,
                'metrics': sim.data_collector.get_kpi_summary()
            }
            self.calibration_results.append(result)
            
            # Update best parameters
            if score < best_score:
                best_score = score
                best_params = params
                self.best_parameters = params
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{num_iterations}, Best Score: {best_score:.4f}")
        
        print(f"✅ Calibration complete. Best score: {best_score:.4f}")
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': self.calibration_results
        }
    
    def _sample_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a random parameter combination"""
        params = {}
        
        # Sample agent attributes
        for attr, values in parameter_space['agent_attributes'].items():
            params[attr] = np.random.choice(values)
        
        # Sample UA channel parameters
        for channel_param, values in parameter_space['ua_channels'].items():
            params[channel_param] = np.random.choice(values)
        
        return params
    
    def _create_simulation_with_params(self, params: Dict[str, Any], num_agents: int) -> MarketingSimulation:
        """Create simulation with specific parameters"""
        # Create config dictionary
        config = {
            'agent_attributes': {},
            'ua_channels': {
                'paid_social': {},
                'video_ads': {},
                'search_ads': {},
                'owned_channels': {}
            }
        }
        
        # Map parameters to config
        agent_mapping = {
            'install_threshold': 'install_threshold_range',
            'iap_probability': 'iap_probability_range',
            'k_factor': 'k_factor_range',
            'churn_propensity': 'churn_propensity_range'
        }
        
        for param, value in params.items():
            if param in agent_mapping:
                config['agent_attributes'][agent_mapping[param]] = (value * 0.8, value * 1.2)
            elif 'paid_social' in param:
                key = param.replace('paid_social_', '')
                config['ua_channels']['paid_social'][key] = value
            elif 'video_ads' in param:
                key = param.replace('video_ads_', '')
                config['ua_channels']['video_ads'][key] = value
            elif 'search_ads' in param:
                key = param.replace('search_ads_', '')
                config['ua_channels']['search_ads'][key] = value
        
        sim = MarketingSimulation(num_agents=num_agents, config=config)
        return sim
    
    def _calculate_calibration_score(self, sim: MarketingSimulation) -> float:
        """Calculate calibration score based on target metrics"""
        metrics = sim.data_collector.get_kpi_summary()
        
        # Calculate weighted error
        errors = []
        weights = {
            'day1_retention': 2.0,
            'day7_retention': 2.5,
            'day30_retention': 1.5,
            'install_rate': 2.0,
            'paying_rate': 1.5,
            'arpu': 1.0,
            'cac': 1.0
        }
        
        for metric, target_value in self.target_metrics.items():
            if metric in metrics:
                actual_value = metrics[metric]
                error = abs(actual_value - target_value) / target_value
                weighted_error = error * weights.get(metric, 1.0)
                errors.append(weighted_error)
        
        return np.mean(errors) if errors else float('inf')
    
    def validate_model(self, validation_data: pd.DataFrame, num_agents: int = 5000) -> Dict[str, Any]:
        """Validate model against historical data"""
        print("🔍 Starting validation process...")
        
        if not self.best_parameters:
            print("⚠️  No calibrated parameters found. Running calibration first...")
            self.run_calibration(num_agents=num_agents)
        
        # Create simulation with best parameters
        sim = self._create_simulation_with_params(self.best_parameters, num_agents)
        
        # Run simulation for validation period
        validation_days = len(validation_data)
        sim.run_model(steps=validation_days)
        
        # Compare results
        simulation_metrics = sim.data_collector.get_kpi_summary()
        
        validation_results = {
            'validation_period_days': validation_days,
            'simulation_metrics': simulation_metrics,
            'historical_metrics': self._extract_historical_metrics(validation_data),
            'accuracy_metrics': self._calculate_accuracy_metrics(simulation_metrics, validation_data)
        }
        
        self.validation_results = validation_results
        
        print(f"✅ Validation complete")
        print(f"  MAPE: {validation_results['accuracy_metrics']['mape']:.2%}")
        print(f"  RMSE: {validation_results['accuracy_metrics']['rmse']:.2f}")
        
        return validation_results
    
    def _extract_historical_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Extract key metrics from historical data"""
        metrics = {}
        
        # Calculate based on available columns
        if 'installs' in historical_data.columns:
            metrics['total_installs'] = historical_data['installs'].sum()
            metrics['avg_daily_installs'] = historical_data['installs'].mean()
        
        if 'revenue' in historical_data.columns:
            metrics['total_revenue'] = historical_data['revenue'].sum()
            metrics['avg_daily_revenue'] = historical_data['revenue'].mean()
        
        if 'spend' in historical_data.columns:
            metrics['total_spend'] = historical_data['spend'].sum()
            metrics['cac'] = metrics['total_spend'] / max(metrics.get('total_installs', 1), 1)
        
        if 'mau' in historical_data.columns:
            metrics['avg_mau'] = historical_data['mau'].mean()
        
        return metrics
    
    def _calculate_accuracy_metrics(self, sim_metrics: Dict[str, float], 
                                  historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate accuracy metrics comparing simulation to historical data"""
        hist_metrics = self._extract_historical_metrics(historical_data)
        
        # Prepare comparison data
        comparison_pairs = []
        
        if 'total_installs' in hist_metrics and 'total_installs' in sim_metrics:
            comparison_pairs.append((hist_metrics['total_installs'], sim_metrics['total_installs']))
        
        if 'total_revenue' in hist_metrics and 'total_revenue' in sim_metrics:
            comparison_pairs.append((hist_metrics['total_revenue'], sim_metrics['total_revenue']))
        
        if 'total_spend' in hist_metrics:
            sim_spend = sim_metrics.get('cac', 0) * sim_metrics.get('total_installs', 0)
            comparison_pairs.append((hist_metrics['total_spend'], sim_spend))
        
        if not comparison_pairs:
            return {'mape': 0.0, 'rmse': 0.0}
        
        # Calculate MAPE and RMSE
        historical_values = [pair[0] for pair in comparison_pairs]
        simulated_values = [pair[1] for pair in comparison_pairs]
        
        # MAPE calculation
        percentage_errors = []
        for hist, sim in comparison_pairs:
            if hist != 0:
                percentage_errors.append(abs(hist - sim) / hist)
        
        mape = np.mean(percentage_errors) if percentage_errors else 0.0
        
        # RMSE calculation
        mse = np.mean([(hist - sim) ** 2 for hist, sim in comparison_pairs])
        rmse = np.sqrt(mse)
        
        return {
            'mape': mape,
            'rmse': rmse,
            'num_metrics_compared': len(comparison_pairs)
        }
    
    def hindcast_test(self, train_period: Tuple[int, int], test_period: Tuple[int, int], 
                     historical_data: pd.DataFrame, num_agents: int = 5000) -> Dict[str, Any]:
        """Perform hindcast test (train on one period, test on another)"""
        print(f"🔮 Running hindcast test: Train days {train_period[0]}-{train_period[1]}, Test days {test_period[0]}-{test_period[1]}")
        
        # Split data
        train_data = historical_data.iloc[train_period[0]:train_period[1]]
        test_data = historical_data.iloc[test_period[0]:test_period[1]]
        
        # Calibrate on training period
        print("  Calibrating on training period...")
        self.run_calibration(num_agents=num_agents, simulation_days=train_period[1] - train_period[0])
        
        # Validate on test period
        print("  Validating on test period...")
        validation_results = self.validate_model(test_data, num_agents)
        
        return {
            'train_period': train_period,
            'test_period': test_period,
            'calibration_results': self.calibration_results[-1],  # Last calibration result
            'validation_results': validation_results
        }
    
    def get_calibration_report(self) -> str:
        """Generate calibration report"""
        if not self.best_parameters:
            return "No calibration results available."
        
        report = "🔧 Calibration Report\n"
        report += "=" * 30 + "\n\n"
        
        report += "Best Parameters:\n"
        for param, value in self.best_parameters.items():
            report += f"  {param}: {value:.4f}\n"
        
        if self.calibration_results:
            best_result = min(self.calibration_results, key=lambda x: x['score'])
            report += f"\nBest Score: {best_result['score']:.4f}\n"
            
            report += "\nAchieved Metrics:\n"
            for metric, value in best_result['metrics'].items():
                report += f"  {metric}: {value:.4f}\n"
        
        if self.validation_results:
            report += f"\nValidation Results:\n"
            accuracy = self.validation_results['accuracy_metrics']
            report += f"  MAPE: {accuracy['mape']:.2%}\n"
            report += f"  RMSE: {accuracy['rmse']:.2f}\n"
            report += f"  Metrics Compared: {accuracy['num_metrics_compared']}\n"
        
        return report
    
    def save_results(self, filepath: str):
        """Save calibration and validation results"""
        results = {
            'best_parameters': self.best_parameters,
            'calibration_results': self.calibration_results,
            'validation_results': self.validation_results,
            'target_metrics': self.target_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load calibration and validation results"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_parameters = results.get('best_parameters', {})
        self.calibration_results = results.get('calibration_results', [])
        self.validation_results = results.get('validation_results', {})
        self.target_metrics = results.get('target_metrics', {})
        
        print(f"Results loaded from {filepath}")


# Utility functions for creating sample historical data
def create_sample_historical_data(days: int = 90) -> pd.DataFrame:
    """Create sample historical data for testing"""
    np.random.seed(42)
    
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    # Generate realistic mobile gaming metrics
    base_installs = 1000
    installs = [base_installs + np.random.normal(0, 100) for _ in range(days)]
    installs = [max(0, int(install)) for install in installs]
    
    base_revenue = 2500
    revenue = [base_revenue + np.random.normal(0, 200) for _ in range(days)]
    revenue = [max(0, revenue_day) for revenue_day in revenue]
    
    base_spend = 3000
    spend = [base_spend + np.random.normal(0, 150) for _ in range(days)]
    spend = [max(0, spend_day) for spend_day in spend]
    
    # MAU with some seasonality
    base_mau = 50000
    seasonality = [5000 * np.sin(2 * np.pi * i / 30) for i in range(days)]
    mau = [base_mau + season + np.random.normal(0, 1000) for i, season in enumerate(seasonality)]
    mau = [max(0, int(mau_value)) for mau_value in mau]
    
    return pd.DataFrame({
        'date': dates,
        'installs': installs,
        'revenue': revenue,
        'spend': spend,
        'mau': mau
    })