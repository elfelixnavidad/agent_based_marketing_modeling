import mesa
from enum import Enum
import random
import numpy as np


class PlayerState(Enum):
    UNAWARE = "unaware"
    AWARE = "aware"
    INSTALLED = "installed"
    PAYING_USER = "paying_user"
    LAPSED = "lapsed"


class PlayerPersona(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = PlayerState.UNAWARE
        
        # Persona attributes
        self.channel_preference = {
            'social': random.uniform(0.1, 1.0),
            'video': random.uniform(0.1, 1.0),
            'search': random.uniform(0.1, 1.0),
            'organic': random.uniform(0.1, 1.0)
        }
        
        self.install_threshold = random.uniform(0.3, 0.8)
        self.iap_probability = random.uniform(0.01, 0.15)
        self.k_factor = random.uniform(0.1, 0.5)
        self.churn_propensity = random.uniform(0.01, 0.05)
        
        # Tracking variables
        self.install_time = None
        self.total_spend = 0.0
        self.days_active = 0
        self.sessions_played = 0
        
    def step(self):
        """Execute one step of the agent's behavior"""
        if self.state == PlayerState.UNAWARE:
            self._handle_unaware_state()
        elif self.state == PlayerState.AWARE:
            self._handle_aware_state()
        elif self.state == PlayerState.INSTALLED:
            self._handle_installed_state()
        elif self.state == PlayerState.PAYING_USER:
            self._handle_paying_user_state()
        elif self.state == PlayerState.LAPSED:
            self._handle_lapsed_state()
    
    def _handle_unaware_state(self):
        """Handle behavior when agent is unaware of the game"""
        # Check if any UA campaigns have reached this agent
        awareness_score = self.model.ua_manager.calculate_awareness_impact(self)
        
        if awareness_score > self.install_threshold:
            self.state = PlayerState.AWARE
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.UNAWARE, PlayerState.AWARE
                )
    
    def _handle_aware_state(self):
        """Handle behavior when agent is aware but hasn't installed"""
        # Check if agent should install based on continued exposure
        install_score = self.model.ua_manager.calculate_install_impact(self)
        
        if install_score > self.install_threshold:
            self.state = PlayerState.INSTALLED
            self.install_time = self.model.schedule.steps
            if hasattr(self.model.data_collector, 'record_install'):
                self.model.data_collector.record_install(self.unique_id)
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.AWARE, PlayerState.INSTALLED
                )
    
    def _handle_installed_state(self):
        """Handle behavior when agent is an active player"""
        self.days_active += 1
        
        # Check for churn
        if random.random() < self._calculate_churn_probability():
            self.state = PlayerState.LAPSED
            if hasattr(self.model.data_collector, 'record_churn'):
                self.model.data_collector.record_churn(self.unique_id)
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.INSTALLED, PlayerState.LAPSED
                )
            return
        
        # Check for IAP conversion
        if random.random() < self.iap_probability:
            self._make_purchase()
        
        # Organic influence (k-factor)
        self._spread_awareness()
        
        # Engage with owned channels
        self.model.ua_manager.handle_owned_channel_interaction(self)
    
    def _handle_paying_user_state(self):
        """Handle behavior when agent is a paying user"""
        self._handle_installed_state()  # Paying users have all installed behaviors
        
        # Higher retention for paying users
        self.churn_propensity *= 0.7
    
    def _handle_lapsed_state(self):
        """Handle behavior when agent has churned"""
        # Small chance of reactivation
        if random.random() < 0.001:  # 0.1% chance per step
            self.state = PlayerState.INSTALLED
            if hasattr(self.model.data_collector, 'record_reactivation'):
                self.model.data_collector.record_reactivation(self.unique_id)
    
    def _calculate_churn_probability(self):
        """Calculate probability of churning based on various factors"""
        base_churn = self.churn_propensity
        
        # Day-based churn curve (higher churn in early days)
        if self.days_active < 7:
            base_churn *= 2.0
        elif self.days_active < 30:
            base_churn *= 1.5
        
        # Reduce churn if engaged with owned channels
        engagement_bonus = self.model.ua_manager.calculate_engagement_bonus(self)
        
        return max(0, base_churn - engagement_bonus)
    
    def _make_purchase(self):
        """Simulate an in-app purchase"""
        if self.state == PlayerState.INSTALLED:
            self.state = PlayerState.PAYING_USER
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.INSTALLED, PlayerState.PAYING_USER
                )
        
        # Purchase amount follows a power law distribution
        purchase_amount = np.random.pareto(1.0) * 2.0 + 0.99  # Min $0.99
        purchase_amount = min(purchase_amount, 99.99)  # Max $99.99
        
        self.total_spend += purchase_amount
        if hasattr(self.model.data_collector, 'record_purchase'):
                self.model.data_collector.record_purchase(self.unique_id, purchase_amount)
    
    def _spread_awareness(self):
        """Spread awareness to other agents (organic growth)"""
        if self.k_factor > 0:
            # Find unaware agents to influence
            unaware_agents = [
                agent for agent in self.model.schedule.agents
                if agent.state == PlayerState.UNAWARE and agent.unique_id != self.unique_id
            ]
            
            if unaware_agents:
                # Number of agents to influence based on k-factor
                num_to_influence = min(int(self.k_factor * 10), len(unaware_agents))
                agents_to_influence = random.sample(unaware_agents, num_to_influence)
                
                for agent in agents_to_influence:
                    # Probability of successful awareness transfer
                    if random.random() < 0.3:  # 30% chance
                        agent.state = PlayerState.AWARE
                        # Record organic conversion if method exists
                        if hasattr(self.model.data_collector, 'record_organic_conversion'):
                            self.model.data_collector.record_organic_conversion(
                                self.unique_id, agent.unique_id
                            )
    
    def get_ltv(self):
        """Calculate lifetime value of this agent"""
        return self.total_spend
    
    def get_retention_days(self):
        """Get number of days agent has been retained"""
        if self.install_time is None:
            return 0
        return self.model.schedule.steps - self.install_time