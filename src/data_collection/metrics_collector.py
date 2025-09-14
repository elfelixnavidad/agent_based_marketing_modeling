import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class MetricsCollector:
    """Collects and manages simulation metrics and data"""
    
    def __init__(self, model):
        self.model = model
        self.current_step = 0
        
        # Time series data
        self.time_series_data = []
        self.daily_metrics = pd.DataFrame()
        
        # Event tracking
        self.state_changes = []
        self.installs = []
        self.churns = []
        self.purchases = []
        self.reactivations = []
        self.organic_conversions = []
        
        # Cohort analysis
        self.cohort_data = {}
        self.retention_curves = {}
        
        # Channel performance
        self.channel_performance = {}
        
        # Initialize metrics structure
        self._initialize_metrics_structure()
    
    def _initialize_metrics_structure(self):
        """Initialize the metrics data structure"""
        self.metrics = {
            'step': [],
            'date': [],
            'total_agents': [],
            'unaware_agents': [],
            'aware_agents': [],
            'installed_agents': [],
            'paying_users': [],
            'lapsed_agents': [],
            'installs_today': [],
            'churns_today': [],
            'purchases_today': [],
            'revenue_today': [],
            'cumulative_revenue': [],
            'arpu': [],
            'arppu': [],
            'daily_budget_spend': [],
            'cumulative_spend': [],
            'roas': [],
            'organic_installs': [],
            'paid_installs': [],
        }
    
    def record_step_metrics(self):
        """Record metrics for the current step"""
        step = self.model.schedule.steps
        date = datetime.now() + timedelta(days=step)
        
        # Count agents by state
        agent_counts = self.model.get_agent_distribution()
        
        # Calculate daily metrics
        daily_installs = len([i for i in self.installs if i['step'] == step])
        daily_churns = len([c for c in self.churns if c['step'] == step])
        daily_purchases = [p for p in self.purchases if p['step'] == step]
        daily_revenue = sum(p['amount'] for p in daily_purchases)
        
        # Get UA performance
        ua_performance = self.model.ua_manager.get_performance_metrics()
        daily_spend = sum(metrics['spend'] for metrics in ua_performance.values())
        
        # Calculate cumulative metrics
        cumulative_revenue = sum(p['amount'] for p in self.purchases)
        cumulative_installs = len(self.installs)
        
        # Calculate rates
        total_active_users = agent_counts.get('installed', 0) + agent_counts.get('paying_user', 0)
        paying_users = agent_counts.get('paying_user', 0)
        
        arpu = cumulative_revenue / max(total_active_users, 1)
        arppu = cumulative_revenue / max(paying_users, 1)
        roas = cumulative_revenue / max(daily_spend, 1) if daily_spend > 0 else 0
        
        # Record metrics
        step_metrics = {
            'step': step,
            'date': date.strftime('%Y-%m-%d'),
            'total_agents': self.model.schedule.get_agent_count(),
            'unaware_agents': agent_counts.get('unaware', 0),
            'aware_agents': agent_counts.get('aware', 0),
            'installed_agents': agent_counts.get('installed', 0),
            'paying_users': agent_counts.get('paying_user', 0),
            'lapsed_agents': agent_counts.get('lapsed', 0),
            'installs_today': daily_installs,
            'churns_today': daily_churns,
            'purchases_today': len(daily_purchases),
            'revenue_today': daily_revenue,
            'cumulative_revenue': cumulative_revenue,
            'arpu': arpu,
            'arppu': arppu,
            'daily_budget_spend': daily_spend,
            'cumulative_spend': sum(c['spend'] for c in ua_performance.values()),
            'roas': roas,
            'organic_installs': len([o for o in self.organic_conversions if o['step'] == step]),
            'paid_installs': daily_installs - len([o for o in self.organic_conversions if o['step'] == step]),
        }
        
        self.time_series_data.append(step_metrics)
        
        # Update daily metrics DataFrame
        if self.daily_metrics.empty:
            self.daily_metrics = pd.DataFrame([step_metrics])
        else:
            self.daily_metrics = pd.concat([self.daily_metrics, pd.DataFrame([step_metrics])], ignore_index=True)
    
    def record_state_change(self, agent_id: int, old_state: str, new_state: str):
        """Record agent state changes"""
        self.state_changes.append({
            'step': self.model.schedule.steps,
            'agent_id': agent_id,
            'old_state': old_state,
            'new_state': new_state,
            'timestamp': datetime.now()
        })
    
    def record_install(self, agent_id: int):
        """Record a new install"""
        self.installs.append({
            'step': self.model.schedule.steps,
            'agent_id': agent_id,
            'timestamp': datetime.now()
        })
        
        # Add to cohort analysis
        self._add_to_cohort(agent_id)
    
    def record_churn(self, agent_id: int):
        """Record a churn event"""
        self.churns.append({
            'step': self.model.schedule.steps,
            'agent_id': agent_id,
            'timestamp': datetime.now()
        })
    
    def record_purchase(self, agent_id: int, amount: float):
        """Record an in-app purchase"""
        self.purchases.append({
            'step': self.model.schedule.steps,
            'agent_id': agent_id,
            'amount': amount,
            'timestamp': datetime.now()
        })
    
    def record_reactivation(self, agent_id: int):
        """Record a reactivation event"""
        self.reactivations.append({
            'step': self.model.schedule.steps,
            'agent_id': agent_id,
            'timestamp': datetime.now()
        })
    
    def record_organic_conversion(self, source_agent_id: int, target_agent_id: int):
        """Record organic conversion from viral effects"""
        self.organic_conversions.append({
            'step': self.model.schedule.steps,
            'source_agent_id': source_agent_id,
            'target_agent_id': target_agent_id,
            'timestamp': datetime.now()
        })
    
    def _add_to_cohort(self, agent_id: int):
        """Add agent to cohort analysis"""
        install_step = self.model.schedule.steps
        
        # Determine cohort (e.g., weekly cohorts)
        cohort_week = install_step // 7
        
        if cohort_week not in self.cohort_data:
            self.cohort_data[cohort_week] = []
        
        self.cohort_data[cohort_week].append({
            'agent_id': agent_id,
            'install_step': install_step,
            'cohort_week': cohort_week
        })
    
    def calculate_retention_curves(self):
        """Calculate retention curves for different cohorts"""
        self.retention_curves = {}
        
        for cohort_week, cohort_agents in self.cohort_data.items():
            retention_data = []
            
            for day in range(0, 90, 7):  # Calculate weekly retention for 90 days
                active_count = 0
                total_count = len(cohort_agents)
                
                if total_count == 0:
                    continue
                
                for agent_data in cohort_agents:
                    agent_id = agent_data['agent_id']
                    install_step = agent_data['install_step']
                    
                    # Check if agent is still active at this day
                    check_step = install_step + day
                    
                    # Find agent and check state
                    agent = None
                    for a in self.model.schedule.agents:
                        if a.unique_id == agent_id:
                            agent = a
                            break
                    
                    if agent and agent.state.value in ['installed', 'paying_user']:
                        active_count += 1
                
                retention_rate = active_count / total_count if total_count > 0 else 0
                retention_data.append({
                    'day': day,
                    'retention_rate': retention_rate,
                    'active_users': active_count,
                    'total_users': total_count
                })
            
            self.retention_curves[cohort_week] = retention_data
    
    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get summary of key performance indicators"""
        if self.daily_metrics.empty:
            return {}
        
        latest_metrics = self.daily_metrics.iloc[-1]
        
        # Calculate LTV (simplified)
        total_revenue = latest_metrics['cumulative_revenue']
        total_installs = len(self.installs)
        ltv = total_revenue / max(total_installs, 1)
        
        # Calculate CAC (Customer Acquisition Cost)
        total_spend = latest_metrics['cumulative_spend']
        cac = total_spend / max(total_installs, 1)
        
        # Calculate retention rates
        day1_retention = self._calculate_retention_rate(1)
        day7_retention = self._calculate_retention_rate(7)
        day30_retention = self._calculate_retention_rate(30)
        
        return {
            'total_revenue': total_revenue,
            'total_installs': total_installs,
            'total_spend': total_spend,
            'ltv': ltv,
            'cac': cac,
            'ltv_cac_ratio': ltv / max(cac, 1),
            'current_mau': latest_metrics['installed_agents'] + latest_metrics['paying_users'],
            'paying_users': latest_metrics['paying_users'],
            'arpu': latest_metrics['arpu'],
            'arppu': latest_metrics['arppu'],
            'day1_retention': day1_retention,
            'day7_retention': day7_retention,
            'day30_retention': day30_retention,
            'organic_percentage': latest_metrics['organic_installs'] / max(latest_metrics['installs_today'], 1),
            'roas': latest_metrics['roas'],
        }
    
    def _calculate_retention_rate(self, days: int) -> float:
        """Calculate retention rate for specific day"""
        if not self.installs:
            return 0
        
        target_step = self.model.schedule.steps - days
        if target_step < 0:
            return 0
        
        # Find agents installed at target step
        agents_installed = [
            install['agent_id'] for install in self.installs 
            if install['step'] == target_step
        ]
        
        if not agents_installed:
            return 0
        
        # Check how many are still active
        active_count = 0
        for agent_id in agents_installed:
            agent = None
            for a in self.model.schedule.agents:
                if a.unique_id == agent_id:
                    agent = a
                    break
            
            if agent and agent.state.value in ['installed', 'paying_user']:
                active_count += 1
        
        return active_count / len(agents_installed)
    
    def get_time_series_data(self) -> pd.DataFrame:
        """Get time series metrics as DataFrame"""
        return self.daily_metrics.copy()
    
    def get_cohort_analysis(self) -> Dict[str, List]:
        """Get cohort analysis data"""
        self.calculate_retention_curves()
        return self.retention_curves
    
    def get_channel_performance(self) -> Dict[str, Any]:
        """Get detailed channel performance metrics"""
        return self.model.ua_manager.get_performance_metrics()
    
    def export_data(self, filepath: str):
        """Export all collected data to JSON file"""
        export_data = {
            'time_series': self.time_series_data,
            'state_changes': [self._serialize_event(event) for event in self.state_changes],
            'installs': [self._serialize_event(event) for event in self.installs],
            'churns': [self._serialize_event(event) for event in self.churns],
            'purchases': [self._serialize_event(event) for event in self.purchases],
            'reactivations': [self._serialize_event(event) for event in self.reactivations],
            'organic_conversions': [self._serialize_event(event) for event in self.organic_conversions],
            'cohort_data': self.cohort_data,
            'kpi_summary': self.get_kpi_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _serialize_event(self, event: Dict) -> Dict:
        """Serialize event for JSON export"""
        serialized = event.copy()
        if 'timestamp' in serialized:
            serialized['timestamp'] = serialized['timestamp'].isoformat()
        return serialized
    
    def get_funnel_analysis(self) -> Dict[str, Any]:
        """Analyze conversion funnel"""
        agent_distribution = self.model.get_agent_distribution()
        
        total_agents = self.model.schedule.get_agent_count()
        
        funnel = {
            'unaware': {
                'count': agent_distribution.get('unaware', 0),
                'percentage': agent_distribution.get('unaware', 0) / max(total_agents, 1) * 100
            },
            'aware': {
                'count': agent_distribution.get('aware', 0),
                'percentage': agent_distribution.get('aware', 0) / max(total_agents, 1) * 100
            },
            'installed': {
                'count': agent_distribution.get('installed', 0),
                'percentage': agent_distribution.get('installed', 0) / max(total_agents, 1) * 100
            },
            'paying_user': {
                'count': agent_distribution.get('paying_user', 0),
                'percentage': agent_distribution.get('paying_user', 0) / max(total_agents, 1) * 100
            },
            'lapsed': {
                'count': agent_distribution.get('lapsed', 0),
                'percentage': agent_distribution.get('lapsed', 0) / max(total_agents, 1) * 100
            }
        }
        
        # Calculate conversion rates
        if funnel['aware']['count'] > 0:
            funnel['aware_to_install_rate'] = (funnel['installed']['count'] + funnel['paying_user']['count']) / funnel['aware']['count'] * 100
        else:
            funnel['aware_to_install_rate'] = 0
        
        if (funnel['installed']['count'] + funnel['paying_user']['count']) > 0:
            funnel['install_to_paying_rate'] = funnel['paying_user']['count'] / (funnel['installed']['count'] + funnel['paying_user']['count']) * 100
        else:
            funnel['install_to_paying_rate'] = 0
        
        return funnel
    
    def collect(self, model):
        """Collect data from model (Mesa DataCollector compatibility)"""
        self.record_step_metrics()