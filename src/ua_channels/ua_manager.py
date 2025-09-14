from typing import Dict, List, Any, Optional
from src.ua_channels.ua_channel import (
    create_channel, get_available_channels, register_channel,
    PaidSocialChannel, VideoAdsChannel, SearchAdsChannel, OwnedChannel,
    InfluencerChannel, OOHChannel, ProgrammaticDisplayChannel, SocialOrganicChannel,
    CHANNEL_REGISTRY
)


class UAManager:
    """Manages all UA channels and coordinates campaigns with enhanced extensibility"""

    def __init__(self, model, custom_channels: Optional[Dict[str, Dict[str, Any]]] = None):
        self.model = model
        self.channels = {}
        self.channel_configs = {}
        self.custom_channels = custom_channels or {}
        self.initialize_channels()

    def initialize_channels(self):
        """Initialize all UA channels with default configurations"""
        # Default configurations for each channel type
        self.channel_configs = {
            'paid_social': {
                'enabled': True,
                'priority': 1,
                'description': 'Facebook, Instagram, Twitter paid campaigns'
            },
            'video_ads': {
                'enabled': True,
                'priority': 2,
                'description': 'TikTok, YouTube, Instagram Reels'
            },
            'search_ads': {
                'enabled': True,
                'priority': 3,
                'description': 'Google Ads, App Store Search'
            },
            'owned_channels': {
                'enabled': True,
                'priority': 10,
                'description': 'Push notifications, email, in-game events'
            },
            'influencer': {
                'enabled': False,
                'priority': 4,
                'description': 'Paid influencer partnerships'
            },
            'ooh': {
                'enabled': False,
                'priority': 5,
                'description': 'Billboards, transit advertising'
            },
            'programmatic_display': {
                'enabled': False,
                'priority': 6,
                'description': 'Programmatic banner and native ads'
            },
            'organic_social': {
                'enabled': True,
                'priority': 8,
                'description': 'Organic social media and content marketing'
            }
        }

        # Apply budget allocations
        budget_allocations = self._get_default_budget_allocations()
        for channel_name, budget in budget_allocations.items():
            if channel_name in self.channel_configs:
                self.channel_configs[channel_name]['initial_budget'] = budget

        # Override with model config if available
        if hasattr(self.model, 'config') and 'ua_channels' in self.model.config:
            model_channel_config = self.model.config['ua_channels']
            for channel_name, config in model_channel_config.items():
                if channel_name in self.channel_configs:
                    self.channel_configs[channel_name].update(config)
                else:
                    # Handle custom channel configurations
                    self.channel_configs[channel_name] = config

        # Apply custom channel configurations
        for channel_name, config in self.custom_channels.items():
            if channel_name in self.channel_configs:
                self.channel_configs[channel_name].update(config)
            else:
                self.channel_configs[channel_name] = config

        # Create channel instances using factory function
        for channel_name, config in self.channel_configs.items():
            if config.get('enabled', True):
                try:
                    self.channels[channel_name] = create_channel(channel_name, self.model, config)
                except ValueError as e:
                    print(f"Warning: Could not create channel {channel_name}: {e}")

    def _get_default_budget_allocations(self) -> Dict[str, float]:
        """Get default budget allocations for channels"""
        total_budget = getattr(self.model, 'initial_budget', 200000)

        allocations = {
            'paid_social': total_budget * 0.35,  # 35%
            'video_ads': total_budget * 0.25,    # 25%
            'search_ads': total_budget * 0.20,   # 20%
            'owned_channels': 0,                 # No acquisition budget
            'influencer': total_budget * 0.10,   # 10%
            'ooh': total_budget * 0.05,          # 5%
            'programmatic_display': total_budget * 0.05,  # 5%
            'organic_social': total_budget * 0.05        # 5% for content creation
        }

        return allocations

    def add_channel(self, channel_name: str, channel_config: Dict[str, Any]):
        """Add a new channel to the manager"""
        self.channel_configs[channel_name] = channel_config

        if channel_config.get('enabled', True):
            try:
                self.channels[channel_name] = create_channel(channel_name, self.model, channel_config)
            except ValueError as e:
                print(f"Warning: Could not create channel {channel_name}: {e}")

    def remove_channel(self, channel_name: str):
        """Remove a channel from the manager"""
        if channel_name in self.channels:
            del self.channels[channel_name]
        if channel_name in self.channel_configs:
            del self.channel_configs[channel_name]

    def enable_channel(self, channel_name: str):
        """Enable a disabled channel"""
        if channel_name in self.channel_configs:
            self.channel_configs[channel_name]['enabled'] = True
            if channel_name not in self.channels:
                self.add_channel(channel_name, self.channel_configs[channel_name])

    def disable_channel(self, channel_name: str):
        """Disable a channel"""
        if channel_name in self.channel_configs:
            self.channel_configs[channel_name]['enabled'] = False
        if channel_name in self.channels:
            del self.channels[channel_name]

    def get_channel_by_type(self, channel_type: str) -> Optional[Any]:
        """Get channel by type name"""
        return self.channels.get(channel_type)

    def get_channels_by_type(self, channel_type: str) -> List[Any]:
        """Get all channels of a specific type"""
        return [channel for channel in self.channels.values() if channel.channel_type.value == channel_type]

    def get_enabled_channels(self) -> Dict[str, Any]:
        """Get all enabled channels"""
        return {name: channel for name, channel in self.channels.items() if channel.enabled}

    def get_channel_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive summary of all channel activities"""
        summary = {}

        for name, channel in self.channels.items():
            metrics = channel.get_performance_metrics()
            summary[name] = {
                'name': channel.name,
                'type': channel.channel_type.value,
                'description': channel.description,
                'enabled': channel.enabled,
                'priority': channel.priority,
                'budget': channel.budget,
                'spent': channel.total_spend,
                'remaining': channel.budget - channel.total_spend,
                'impressions': channel.impressions,
                'clicks': channel.clicks,
                'installs': channel.installs,
                'cpi': metrics.get('cpi', 0),
                'ctr': metrics.get('ctr', 0),
                'conversion_rate': metrics.get('conversion_rate', 0),
                'roas': metrics.get('roas', 0),
                'is_exhausted': channel.is_budget_exhausted(),
                'target_states': channel.target_states,
                'custom_metrics': channel.custom_metrics
            }

        return summary

    def update_campaigns(self, priority_order: bool = True) -> Dict[str, Dict[str, Any]]:
        """Update all campaigns for the current step with optional priority ordering"""
        agents = list(self.model.schedule.agents)
        campaign_results = {}

        # Get channels in priority order if requested
        channels_to_process = self.channels.items()
        if priority_order:
            channels_to_process = sorted(self.channels.items(), key=lambda x: x[1].priority)

        for channel_name, channel in channels_to_process:
            if channel.enabled and not channel.is_budget_exhausted():
                results = channel.execute_campaign(agents)
                campaign_results[channel_name] = results

        return campaign_results

    def calculate_awareness_impact(self, agent) -> float:
        """Calculate awareness impact on an agent from all acquisition channels"""
        total_awareness = 0

        for channel_name, channel in self.channels.items():
            # Only paid acquisition and earned media channels create awareness
            if (channel.channel_type.value in ['paid_acquisition', 'earned_media'] and
                channel.enabled and not channel.is_budget_exhausted()):

                # Base awareness from channel activity
                channel_awareness = 0.1 * channel.effectiveness_multiplier

                # Modify by agent's channel preference
                preference_multiplier = agent.channel_preference.get(channel.channel_preference_key, 0.5)

                total_awareness += channel_awareness * preference_multiplier

        return min(total_awareness, 1.0)  # Cap at 1.0

    def calculate_install_impact(self, agent) -> float:
        """Calculate install impact on an agent from all channels"""
        total_install_impact = 0

        for channel_name, channel in self.channels.items():
            # Only acquisition channels drive installs directly
            if (channel.channel_type.value in ['paid_acquisition', 'earned_media'] and
                channel.enabled and not channel.is_budget_exhausted() and
                agent.state.value == 'aware'):

                # Base install impact
                channel_impact = 0.15 * channel.effectiveness_multiplier

                # Modify by agent's channel preference
                preference_multiplier = agent.channel_preference.get(channel.channel_preference_key, 0.5)

                total_install_impact += channel_impact * preference_multiplier

        return min(total_install_impact, 1.0)  # Cap at 1.0

    def handle_owned_channel_interaction(self, agent):
        """Handle owned channel interactions for an agent"""
        for channel_name, channel in self.channels.items():
            if (channel.channel_type.value == 'owned_media' and
                channel.enabled and
                agent.state.value in channel.target_states):
                channel.execute_campaign([agent])

    def calculate_engagement_bonus(self, agent) -> float:
        """Calculate engagement bonus from owned channels"""
        engagement_bonus = 0

        for channel_name, channel in self.channels.items():
            if (channel.channel_type.value == 'owned_media' and
                channel.enabled and
                hasattr(channel, 'retention_boost')):
                engagement_bonus += channel.retention_boost * 0.1  # Small bonus per interaction

        return engagement_bonus

    def set_channel_budget(self, channel_name: str, budget: float):
        """Set budget for a specific channel"""
        if channel_name in self.channels:
            self.channels[channel_name].budget = budget
        elif channel_name in self.channel_configs:
            self.channel_configs[channel_name]['initial_budget'] = budget

    def get_channel_budgets(self) -> Dict[str, float]:
        """Get current budgets for all channels"""
        return {name: channel.budget for name, channel in self.channels.items()}

    def get_total_spend(self) -> float:
        """Get total spend across all channels"""
        return sum(channel.total_spend for channel in self.channels.values())

    def get_total_budget(self) -> float:
        """Get total budget across all channels"""
        return sum(channel.budget for channel in self.channels.values())

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all channels"""
        return {name: channel.get_performance_metrics() for name, channel in self.channels.items()}

    def get_channels_by_performance(self, metric: str = 'roas', descending: bool = True) -> List[tuple]:
        """Get channels sorted by performance metric"""
        performance_data = []
        for name, channel in self.channels.items():
            metrics = channel.get_performance_metrics()
            performance_data.append((name, metrics.get(metric, 0), channel))

        return sorted(performance_data, key=lambda x: x[1], reverse=descending)

    def reallocate_budget(self, new_budget_allocation: Dict[str, float]):
        """Reallocate budget across channels"""
        for channel_name, new_budget in new_budget_allocation.items():
            self.set_channel_budget(channel_name, new_budget)

    def optimize_budget_allocation(self, target_metric: str = 'roas', min_budget_per_channel: float = 1000) -> Dict[str, float]:
        """Advanced budget optimization based on performance with constraints"""
        performance_metrics = self.get_performance_metrics()

        # Calculate current performance scores for paid acquisition channels
        channel_scores = {}
        for name, metrics in performance_metrics.items():
            channel = self.channels.get(name)
            if (channel and channel.channel_type.value == 'paid_acquisition' and
                channel.enabled and not channel.is_budget_exhausted()):

                score = metrics.get(target_metric, 0)
                if score > 0:
                    channel_scores[name] = score

        if not channel_scores:
            return self.get_channel_budgets()

        # Calculate proportional allocation based on performance
        total_score = sum(channel_scores.values())
        if total_score == 0:
            return self.get_channel_budgets()

        # Get total available budget for paid channels
        total_available = sum(channel.budget for channel in self.channels.values()
                             if channel.channel_type.value == 'paid_acquisition')

        # Allocate minimum budget to each channel first
        min_allocation = min_budget_per_channel * len(channel_scores)
        remaining_budget = max(0, total_available - min_allocation)

        # Allocate remaining budget proportionally to performance
        optimized_allocation = {}
        for name in channel_scores.keys():
            optimized_allocation[name] = min_budget_per_channel

        if remaining_budget > 0:
            for name, score in channel_scores.items():
                proportion = score / total_score
                optimized_allocation[name] += remaining_budget * proportion

        return optimized_allocation

    def set_daily_spend_limits(self, limits: Dict[str, float]):
        """Set daily spend limits for channels"""
        for channel_name, limit in limits.items():
            if channel_name in self.channels:
                self.channels[channel_name].daily_spend_limit = limit

    def clone_channel(self, source_channel_name: str, new_channel_name: str, new_config: Dict[str, Any] = None):
        """Clone an existing channel with new configuration"""
        if source_channel_name in self.channels:
            source_channel = self.channels[source_channel_name]
            new_channel = source_channel.clone(new_channel_name, new_config)
            self.channels[new_channel_name] = new_channel
            self.channel_configs[new_channel_name] = new_channel.config
            return new_channel
        return None

    def get_channel_insights(self) -> Dict[str, Dict[str, Any]]:
        """Get insights about channel performance and recommendations"""
        insights = {}

        for name, channel in self.channels.items():
            metrics = channel.get_performance_metrics()
            insights[name] = {
                'performance_grade': self._calculate_performance_grade(metrics),
                'budget_efficiency': self._calculate_budget_efficiency(channel, metrics),
                'recommendations': self._generate_channel_recommendations(channel, metrics),
                'risk_factors': self._assess_channel_risks(channel, metrics)
            }

        return insights

    def _calculate_performance_grade(self, metrics: Dict[str, float]) -> str:
        """Calculate performance grade based on metrics"""
        roas = metrics.get('roas', 0)
        if roas >= 3.0:
            return 'A'
        elif roas >= 2.0:
            return 'B'
        elif roas >= 1.0:
            return 'C'
        else:
            return 'D'

    def _calculate_budget_efficiency(self, channel, metrics: Dict[str, float]) -> float:
        """Calculate budget efficiency (ROI per dollar spent)"""
        if channel.total_spend == 0:
            return 0
        return metrics.get('roas', 0)

    def _generate_channel_recommendations(self, channel, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for channel optimization"""
        recommendations = []

        if channel.is_budget_exhausted():
            recommendations.append("Consider increasing budget for this high-performing channel")

        if metrics.get('ctr', 0) < 0.01:
            recommendations.append("Low click-through rate - consider creative optimization")

        if metrics.get('conversion_rate', 0) < 0.05:
            recommendations.append("Low conversion rate - review targeting and landing pages")

        if metrics.get('cpi', 0) > channel.cpi * 1.5:
            recommendations.append("Higher than expected CPI - review bid strategy")

        return recommendations

    def _assess_channel_risks(self, channel, metrics: Dict[str, float]) -> List[str]:
        """Assess potential risks for the channel"""
        risks = []

        if channel.budget < 1000:
            risks.append("Low budget may limit effectiveness")

        if metrics.get('ctr', 0) < 0.005:
            risks.append("Very low engagement - risk of wasted spend")

        if channel.saturation_point and channel.total_spend > channel.saturation_point:
            risks.append("Approaching saturation point - diminishing returns likely")

        return risks

    def reset_daily_metrics(self):
        """Reset daily metrics for all channels"""
        for channel in self.channels.values():
            channel.reset_daily_metrics()

    def get_organic_lift_total(self) -> float:
        """Calculate total organic lift from paid campaigns"""
        total_lift = 0
        for name, channel in self.channels.items():
            if channel.channel_type.value == 'paid_acquisition':
                organic_installs = channel.installs * (channel.organic_lift_factor - 1)
                total_lift += organic_installs

        return total_lift

    def get_channel_mix_analysis(self) -> Dict[str, Any]:
        """Analyze the current channel mix and diversification"""
        analysis = {
            'total_channels': len(self.channels),
            'enabled_channels': len([c for c in self.channels.values() if c.enabled]),
            'channel_types': {},
            'budget_distribution': {},
            'performance_distribution': {},
            'diversification_score': 0
        }

        # Analyze channel types
        for channel in self.channels.values():
            channel_type = channel.channel_type.value
            if channel_type not in analysis['channel_types']:
                analysis['channel_types'][channel_type] = 0
            analysis['channel_types'][channel_type] += 1

        # Budget distribution
        total_budget = self.get_total_budget()
        for name, channel in self.channels.items():
            if channel.enabled:
                percentage = (channel.budget / total_budget * 100) if total_budget > 0 else 0
                analysis['budget_distribution'][name] = percentage

        # Calculate diversification score (0-1, higher is more diversified)
        num_types = len(analysis['channel_types'])
        analysis['diversification_score'] = min(1.0, num_types / 4.0)  # Max score for 4+ types

        return analysis

    def export_configuration(self) -> Dict[str, Any]:
        """Export current UA configuration for replication"""
        return {
            'channel_configs': self.channel_configs,
            'available_channels': get_available_channels(),
            'total_budget': self.get_total_budget(),
            'channel_summary': self.get_channel_summary()
        }