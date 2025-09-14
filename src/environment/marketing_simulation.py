import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

from src.agents.player_persona import PlayerPersona, PlayerState
from src.ua_channels.ua_manager import UAManager
from src.data_collection.metrics_collector import MetricsCollector


class MarketingSimulation(mesa.Model):
    def __init__(self, num_agents=10000, width=100, height=100, 
                 initial_budget=100000, config=None):
        super().__init__()
        
        self.num_agents = num_agents
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.config = config or {}
        
        # Mesa components
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        
        # Initialize subsystems
        self.ua_manager = UAManager(self)
        self.data_collector = MetricsCollector(self)
        
        # Create agents
        self._create_agents()
        
        # Initialize data collection
        self._setup_data_collection()
    
    def _create_agents(self):
        """Create initial population of agents"""
        for i in range(self.num_agents):
            # Random position on grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            
            # Create agent
            agent = PlayerPersona(i, self)
            
            # Add to grid and schedule
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
    
    def _setup_data_collection(self):
        """Set up data collection for model metrics"""
        self.mesa_data_collector = DataCollector(
            model_reporters={
                "Total_Agents": lambda m: m.schedule.get_agent_count(),
                "Unaware_Agents": lambda m: self._count_agents_by_state(PlayerState.UNAWARE),
                "Aware_Agents": lambda m: self._count_agents_by_state(PlayerState.AWARE),
                "Installed_Agents": lambda m: self._count_agents_by_state(PlayerState.INSTALLED),
                "Paying_Users": lambda m: self._count_agents_by_state(PlayerState.PAYING_USER),
                "Lapsed_Agents": lambda m: self._count_agents_by_state(PlayerState.LAPSED),
                "Install_Rate": lambda m: self._calculate_install_rate(),
                "Paying_Rate": lambda m: self._calculate_paying_rate(),
                "Total_Revenue": lambda m: self._calculate_total_revenue(),
                "ARPU": lambda m: self._calculate_arpu(),
                "ARPPU": lambda m: self._calculate_arppu(),
                "Active_Budget": lambda m: m.current_budget,
            },
            agent_reporters={
                "State": lambda a: a.state.value,
                "Total_Spend": lambda a: a.total_spend,
                "Days_Active": lambda a: a.days_active,
            }
        )
    
    def _count_agents_by_state(self, state):
        """Count agents in a specific state"""
        return sum(1 for agent in self.schedule.agents if agent.state == state)
    
    def _calculate_install_rate(self):
        """Calculate current install rate"""
        total_agents = self.schedule.get_agent_count()
        if total_agents == 0:
            return 0
        installed = self._count_agents_by_state(PlayerState.INSTALLED)
        paying = self._count_agents_by_state(PlayerState.PAYING_USER)
        return (installed + paying) / total_agents
    
    def _calculate_paying_rate(self):
        """Calculate paying user conversion rate"""
        installed = self._count_agents_by_state(PlayerState.INSTALLED)
        paying = self._count_agents_by_state(PlayerState.PAYING_USER)
        total_installed = installed + paying
        if total_installed == 0:
            return 0
        return paying / total_installed
    
    def _calculate_total_revenue(self):
        """Calculate total revenue from all agents"""
        return sum(agent.total_spend for agent in self.schedule.agents)
    
    def _calculate_arpu(self):
        """Calculate Average Revenue Per User"""
        installed = self._count_agents_by_state(PlayerState.INSTALLED)
        paying = self._count_agents_by_state(PlayerState.PAYING_USER)
        total_users = installed + paying
        if total_users == 0:
            return 0
        return self._calculate_total_revenue() / total_users
    
    def _calculate_arppu(self):
        """Calculate Average Revenue Per Paying User"""
        paying = self._count_agents_by_state(PlayerState.PAYING_USER)
        if paying == 0:
            return 0
        return self._calculate_total_revenue() / paying
    
    def step(self):
        """Advance the simulation by one step"""
        # Update UA campaigns
        self.ua_manager.update_campaigns()
        
        # Step all agents
        self.schedule.step()
        
        # Collect data
        self.data_collector.collect(self)  # Our MetricsCollector
        self.mesa_data_collector.collect(self)  # Mesa DataCollector
        
        # Update budget based on spend
        self.current_budget -= self.ua_manager.get_total_spend()
    
    def run_model(self, steps=365):
        """Run the simulation for a specified number of steps"""
        for _ in range(steps):
            self.step()
            
            # Stop if budget is exhausted
            if self.current_budget <= 0:
                print(f"Budget exhausted at step {self.schedule.steps}")
                break
    
    def get_summary_stats(self):
        """Get summary statistics for the current state"""
        return {
            "Step": self.schedule.steps,
            "Total_Agents": self.schedule.get_agent_count(),
            "Unaware": self._count_agents_by_state(PlayerState.UNAWARE),
            "Aware": self._count_agents_by_state(PlayerState.AWARE),
            "Installed": self._count_agents_by_state(PlayerState.INSTALLED),
            "Paying_Users": self._count_agents_by_state(PlayerState.PAYING_USER),
            "Lapsed": self._count_agents_by_state(PlayerState.LAPSED),
            "Total_Revenue": self._calculate_total_revenue(),
            "ARPU": self._calculate_arpu(),
            "ARPPU": self._calculate_arppu(),
            "Remaining_Budget": self.current_budget,
        }
    
    def get_agent_distribution(self):
        """Get detailed agent distribution by state"""
        states = [PlayerState.UNAWARE, PlayerState.AWARE, PlayerState.INSTALLED, 
                 PlayerState.PAYING_USER, PlayerState.LAPSED]
        return {state.value: self._count_agents_by_state(state) for state in states}
    
    def set_ua_budget(self, channel_name, budget):
        """Set budget for a specific UA channel"""
        self.ua_manager.set_channel_budget(channel_name, budget)
    
    def get_ua_performance(self):
        """Get performance metrics for all UA channels"""
        return self.ua_manager.get_performance_metrics()