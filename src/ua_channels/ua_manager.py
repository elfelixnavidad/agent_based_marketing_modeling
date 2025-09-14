from typing import Dict, List, Any
from src.ua_channels.ua_channel import PaidSocialChannel, VideoAdsChannel, SearchAdsChannel, OwnedChannel


class UAManager:
    """Manages all UA channels and coordinates campaigns"""
    
    def __init__(self, model):
        self.model = model
        self.channels = {}
        self.initialize_channels()
    
    def initialize_channels(self):
        """Initialize all UA channels with default configurations"""
        # Default configurations for each channel
        channel_configs = {
            'paid_social': {
                'initial_budget': 50000,
                'cpi': 2.5,
                'ctr': 0.03,
                'conversion_rate': 0.08,
                'saturation_point': 8000,
                'effectiveness_multiplier': 1.0,
                'organic_lift_factor': 1.3
            },
            'video_ads': {
                'initial_budget': 30000,
                'cpi': 3.0,
                'ctr': 0.025,
                'conversion_rate': 0.06,
                'saturation_point': 12000,
                'effectiveness_multiplier': 1.1,
                'organic_lift_factor': 1.4
            },
            'search_ads': {
                'initial_budget': 20000,
                'cpi': 4.0,
                'ctr': 0.05,
                'conversion_rate': 0.12,
                'saturation_point': 5000,
                'effectiveness_multiplier': 1.2,
                'organic_lift_factor': 1.1
            },
            'owned_channels': {
                'initial_budget': 0,
                'ctr': 0.15,
                'retention_boost': 0.1,
                'reactivation_rate': 0.02
            }
        }
        
        # Override with model config if available
        if hasattr(self.model, 'config') and 'ua_channels' in self.model.config:
            channel_configs.update(self.model.config['ua_channels'])
        
        # Create channel instances
        self.channels['paid_social'] = PaidSocialChannel(self.model, channel_configs['paid_social'])
        self.channels['video_ads'] = VideoAdsChannel(self.model, channel_configs['video_ads'])
        self.channels['search_ads'] = SearchAdsChannel(self.model, channel_configs['search_ads'])
        self.channels['owned_channels'] = OwnedChannel(self.model, channel_configs['owned_channels'])
    
    def update_campaigns(self):
        """Update all campaigns for the current step"""
        agents = list(self.model.schedule.agents)
        campaign_results = {}
        
        for channel_name, channel in self.channels.items():
            results = channel.execute_campaign(agents)
            campaign_results[channel_name] = results
        
        return campaign_results
    
    def calculate_awareness_impact(self, agent):
        """Calculate awareness impact on an agent from all channels"""
        total_awareness = 0
        
        for channel_name, channel in self.channels.items():
            if channel_name == 'owned_channels':
                continue  # Owned channels don't create awareness
            
            if channel.budget > 0:
                # Base awareness from channel activity
                channel_awareness = 0.1 * channel.effectiveness_multiplier
                
                # Modify by agent's channel preference
                preference_multiplier = agent.channel_preference.get(channel_name, 0.5)
                
                total_awareness += channel_awareness * preference_multiplier
        
        return min(total_awareness, 1.0)  # Cap at 1.0
    
    def calculate_install_impact(self, agent):
        """Calculate install impact on an agent from all channels"""
        total_install_impact = 0
        
        for channel_name, channel in self.channels.items():
            if channel_name == 'owned_channels':
                continue  # Owned channels don't drive installs directly
            
            if channel.budget > 0 and agent.state.value == 'aware':
                # Base install impact
                channel_impact = 0.15 * channel.effectiveness_multiplier
                
                # Modify by agent's channel preference
                preference_multiplier = agent.channel_preference.get(channel_name, 0.5)
                
                total_install_impact += channel_impact * preference_multiplier
        
        return min(total_install_impact, 1.0)  # Cap at 1.0
    
    def handle_owned_channel_interaction(self, agent):
        """Handle owned channel interactions for an agent"""
        if 'owned_channels' in self.channels:
            self.channels['owned_channels'].execute_campaign([agent])
    
    def calculate_engagement_bonus(self, agent):
        """Calculate engagement bonus from owned channels"""
        if 'owned_channels' not in self.channels:
            return 0
        
        owned_channel = self.channels['owned_channels']
        return owned_channel.retention_boost * 0.1  # Small bonus per interaction
    
    def set_channel_budget(self, channel_name, budget):
        """Set budget for a specific channel"""
        if channel_name in self.channels:
            self.channels[channel_name].budget = budget
    
    def get_channel_budgets(self):
        """Get current budgets for all channels"""
        return {name: channel.budget for name, channel in self.channels.items()}
    
    def get_total_spend(self):
        """Get total spend across all channels"""
        return sum(channel.total_spend for channel in self.channels.values())
    
    def get_performance_metrics(self):
        """Get performance metrics for all channels"""
        return {name: channel.get_performance_metrics() for name, channel in self.channels.items()}
    
    def get_channel_summary(self):
        """Get summary of all channel activities"""
        summary = {}
        
        for name, channel in self.channels.items():
            summary[name] = {
                'budget': channel.budget,
                'spent': channel.total_spend,
                'remaining': channel.budget - channel.total_spend,
                'impressions': channel.impressions,
                'clicks': channel.clicks,
                'installs': channel.installs,
                'cpi': channel.total_cost / max(channel.installs, 1),
                'ctr': channel.clicks / max(channel.impressions, 1),
            }
        
        return summary
    
    def reallocate_budget(self, new_budget_allocation):
        """Reallocate budget across channels"""
        total_budget = sum(new_budget_allocation.values())
        
        for channel_name, new_budget in new_budget_allocation.items():
            if channel_name in self.channels:
                self.channels[channel_name].budget = new_budget
    
    def optimize_budget_allocation(self, target_metric='roas'):
        """Simple budget optimization based on performance"""
        performance_metrics = self.get_performance_metrics()
        
        # Calculate current performance scores
        channel_scores = {}
        for name, metrics in performance_metrics.items():
            if name == 'owned_channels':
                continue
            
            score = metrics.get(target_metric, 0)
            if score > 0:
                channel_scores[name] = score
        
        if not channel_scores:
            return self.get_channel_budgets()
        
        # Calculate proportional allocation based on performance
        total_score = sum(channel_scores.values())
        if total_score == 0:
            return self.get_channel_budgets()
        
        # Get total available budget
        total_available = sum(channel.budget for channel in self.channels.values() 
                              if channel.name != 'owned_channels')
        
        # Allocate budget proportionally to performance
        optimized_allocation = {}
        for name, score in channel_scores.items():
            proportion = score / total_score
            optimized_allocation[name] = total_available * proportion
        
        return optimized_allocation
    
    def reset_daily_metrics(self):
        """Reset daily metrics for all channels"""
        for channel in self.channels.values():
            channel.reset_daily_metrics()
    
    def get_organic_lift_total(self):
        """Calculate total organic lift from paid campaigns"""
        total_lift = 0
        for name, channel in self.channels.items():
            if name != 'owned_channels':
                organic_installs = channel.installs * (channel.organic_lift_factor - 1)
                total_lift += organic_installs
        
        return total_lift