import random
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from src.agents.player_persona import PlayerState


class UAChannel(ABC):
    """Abstract base class for UA channels"""
    
    def __init__(self, name: str, model, config: Dict[str, Any] = None):
        self.name = name
        self.model = model
        self.config = config or {}
        
        # Channel metrics
        self.budget = self.config.get('initial_budget', 0)
        self.total_spend = 0
        self.impressions = 0
        self.clicks = 0
        self.installs = 0
        self.total_cost = 0
        
        # Performance metrics
        self.cpi = self.config.get('cpi', 1.0)
        self.ctr = self.config.get('ctr', 0.02)  # Click-through rate
        self.conversion_rate = self.config.get('conversion_rate', 0.05)
        self.effectiveness_multiplier = self.config.get('effectiveness_multiplier', 1.0)
        self.organic_lift_factor = self.config.get('organic_lift_factor', 1.2)
        
        # Saturation and diminishing returns
        self.saturation_point = self.config.get('saturation_point', 10000)
        self.diminishing_returns_rate = self.config.get('diminishing_returns_rate', 0.1)
    
    @abstractmethod
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute campaign for the current step"""
        pass
    
    def calculate_effectiveness(self, spend_amount: float) -> float:
        """Calculate effectiveness considering diminishing returns"""
        if spend_amount <= 0:
            return 0
        
        # Apply diminishing returns
        saturation_factor = 1 - math.exp(-spend_amount / self.saturation_point)
        effectiveness = self.effectiveness_multiplier * saturation_factor
        
        return effectiveness
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this channel"""
        return {
            'budget': self.budget,
            'spend': self.total_spend,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'installs': self.installs,
            'cpi': self.total_cost / max(self.installs, 1),
            'ctr': self.clicks / max(self.impressions, 1),
            'conversion_rate': self.installs / max(self.clicks, 1),
            'roas': self._calculate_roas(),
        }
    
    def _calculate_roas(self) -> float:
        """Calculate Return on Ad Spend"""
        if self.total_cost == 0:
            return 0
        
        # Estimate revenue from installs (rough approximation)
        estimated_revenue = self.installs * 5.0  # Assume $5 LTV per install
        return estimated_revenue / self.total_cost
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.impressions = 0
        self.clicks = 0
        self.installs = 0
        self.daily_spend = 0


class PaidSocialChannel(UAChannel):
    """Paid social media advertising (Facebook, Instagram, etc.)"""
    
    def __init__(self, model, config: Dict[str, Any] = None):
        super().__init__('paid_social', model, config)
        self.cpi = self.config.get('cpi', 2.5)
        self.ctr = self.config.get('ctr', 0.03)
        self.conversion_rate = self.config.get('conversion_rate', 0.08)
        self.saturation_point = self.config.get('saturation_point', 8000)
    
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute paid social campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0
        }
        
        if self.budget <= 0:
            return results
        
        # Calculate daily spend budget
        daily_budget = min(self.budget * 0.1, self.budget)  # Spend 10% per day
        
        # Get unaware and aware agents
        target_agents = [
            agent for agent in agents
            if agent.state.value in ['unaware', 'aware']
        ]
        
        if not target_agents:
            return results
        
        # Calculate effectiveness
        effectiveness = self.calculate_effectiveness(daily_budget)
        
        # Generate impressions
        impressions_per_dollar = 100  # Base impressions per dollar
        impressions = int(daily_budget * impressions_per_dollar * effectiveness)
        results['impressions'] = impressions
        
        # Randomly assign impressions to agents
        agents_reached = random.sample(
            target_agents, 
            min(impressions, len(target_agents))
        )
        
        # Calculate clicks based on CTR and agent preferences
        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get('social', 0.5):
                results['clicks'] += 1
                
                # Calculate conversion to install
                if random.random() < self.conversion_rate * effectiveness:
                    results['installs'] += 1
                    
                    # Convert agent if unaware
                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps
        
        # Calculate costs
        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click
        
        # Update metrics
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']
        
        return results


class VideoAdsChannel(UAChannel):
    """Video advertising (TikTok, YouTube, etc.)"""
    
    def __init__(self, model, config: Dict[str, Any] = None):
        super().__init__('video_ads', model, config)
        self.cpi = self.config.get('cpi', 3.0)
        self.ctr = self.config.get('ctr', 0.025)
        self.conversion_rate = self.config.get('conversion_rate', 0.06)
        self.saturation_point = self.config.get('saturation_point', 12000)
    
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute video ads campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0
        }
        
        if self.budget <= 0:
            return results
        
        daily_budget = min(self.budget * 0.08, self.budget)  # Spend 8% per day
        
        target_agents = [
            agent for agent in agents
            if agent.state.value in ['unaware', 'aware']
        ]
        
        if not target_agents:
            return results
        
        effectiveness = self.calculate_effectiveness(daily_budget)
        
        # Video ads have higher impressions per dollar
        impressions_per_dollar = 150
        impressions = int(daily_budget * impressions_per_dollar * effectiveness)
        results['impressions'] = impressions
        
        agents_reached = random.sample(
            target_agents, 
            min(impressions, len(target_agents))
        )
        
        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get('video', 0.5):
                results['clicks'] += 1
                
                if random.random() < self.conversion_rate * effectiveness:
                    results['installs'] += 1
                    
                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps
        
        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click
        
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']
        
        return results


class SearchAdsChannel(UAChannel):
    """Search advertising (Google Ads, App Store Optimization)"""
    
    def __init__(self, model, config: Dict[str, Any] = None):
        super().__init__('search_ads', model, config)
        self.cpi = self.config.get('cpi', 4.0)
        self.ctr = self.config.get('ctr', 0.05)
        self.conversion_rate = self.config.get('conversion_rate', 0.12)
        self.saturation_point = self.config.get('saturation_point', 5000)
    
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute search ads campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0
        }
        
        if self.budget <= 0:
            return results
        
        daily_budget = min(self.budget * 0.05, self.budget)  # Spend 5% per day
        
        target_agents = [
            agent for agent in agents
            if agent.state.value in ['unaware', 'aware']
        ]
        
        if not target_agents:
            return results
        
        effectiveness = self.calculate_effectiveness(daily_budget)
        
        # Search has lower impressions but higher intent
        impressions_per_dollar = 50
        impressions = int(daily_budget * impressions_per_dollar * effectiveness)
        results['impressions'] = impressions
        
        agents_reached = random.sample(
            target_agents, 
            min(impressions, len(target_agents))
        )
        
        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get('search', 0.5):
                results['clicks'] += 1
                
                # Higher conversion rate due to search intent
                if random.random() < self.conversion_rate * effectiveness * 1.5:
                    results['installs'] += 1
                    
                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps
        
        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click
        
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']
        
        return results


class OwnedChannel(UAChannel):
    """Owned channels (push notifications, email, in-game events)"""
    
    def __init__(self, model, config: Dict[str, Any] = None):
        super().__init__('owned_channels', model, config)
        self.cpi = 0  # No direct acquisition cost
        self.ctr = self.config.get('ctr', 0.15)
        self.retention_boost = self.config.get('retention_boost', 0.1)
        self.reactivation_rate = self.config.get('reactivation_rate', 0.02)
    
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute owned channel campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'reactivations': 0,
            'retention_events': 0,
            'cost': 0
        }
        
        # Target installed and paying users
        target_agents = [
            agent for agent in agents
            if agent.state.value in ['installed', 'paying_user']
        ]
        
        # Also target some lapsed users for reactivation
        lapsed_agents = [
            agent for agent in agents
            if agent.state.value == 'lapsed'
        ]
        
        # Engage active users
        for agent in target_agents:
            if random.random() < self.ctr:
                results['impressions'] += 1
                results['retention_events'] += 1
                
                # Reduce churn probability
                agent.churn_propensity *= (1 - self.retention_boost)
                
                # Increase IAP probability for engaged users
                if random.random() < 0.1:  # 10% chance
                    agent.iap_probability *= 1.2
        
        # Attempt reactivation
        if lapsed_agents and random.random() < self.reactivation_rate:
            reactivate_count = min(int(len(lapsed_agents) * 0.01), 10)
            agents_to_reactivate = random.sample(lapsed_agents, reactivate_count)
            
            for agent in agents_to_reactivate:
                agent.state = PlayerState.INSTALLED
                results['reactivations'] += 1
        
        return results