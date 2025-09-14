import random
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
from enum import Enum
from src.agents.player_persona import PlayerState


class ChannelType(Enum):
    """Enumeration of channel types"""
    PAID_ACQUISITION = "paid_acquisition"
    OWNED_MEDIA = "owned_media"
    EARNED_MEDIA = "earned_media"
    HYBRID = "hybrid"


class UAChannel(ABC):
    """Abstract base class for UA channels with enhanced extensibility"""

    def __init__(self, name: str, channel_type: ChannelType, model, config: Dict[str, Any] = None):
        self.name = name
        self.channel_type = channel_type
        self.model = model
        self.config = config or {}

        # Channel metadata
        self.description = self.config.get('description', '')
        self.enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 1)

        # Core channel metrics
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

        # Target audience settings
        self.target_states = self.config.get('target_states', ['unaware', 'aware'])
        self.channel_preference_key = self.config.get('channel_preference_key', self.name)

        # Daily spend control
        self.daily_spend_limit = self.config.get('daily_spend_limit', None)
        self.daily_spend_rate = self.config.get('daily_spend_rate', 0.1)  # 10% of budget per day

        # Custom metrics (extensible per channel)
        self.custom_metrics = self.config.get('custom_metrics', {})

        # Channel-specific attributes
        self._initialize_channel_specific_attributes()

    def _initialize_channel_specific_attributes(self):
        """Override in subclasses to initialize channel-specific attributes"""
        pass

    @abstractmethod
    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute campaign for the current step"""
        pass

    def get_target_agents(self, agents: List[Any]) -> List[Any]:
        """Get agents that match target states for this channel"""
        return [
            agent for agent in agents
            if agent.state.value in self.target_states
        ]

    def calculate_daily_budget(self) -> float:
        """Calculate daily budget considering limits and spend rate"""
        if self.budget <= 0:
            return 0

        daily_budget = self.budget * self.daily_spend_rate

        # Apply daily spend limit if set
        if self.daily_spend_limit:
            daily_budget = min(daily_budget, self.daily_spend_limit)

        return min(daily_budget, self.budget)
    
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
        metrics = {
            'budget': self.budget,
            'spend': self.total_spend,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'installs': self.installs,
            'cpi': self.total_cost / max(self.installs, 1),
            'ctr': self.clicks / max(self.impressions, 1),
            'conversion_rate': self.installs / max(self.clicks, 1),
            'roas': self._calculate_roas(),
            'channel_type': self.channel_type.value,
            'enabled': self.enabled,
            'priority': self.priority
        }

        # Add custom metrics
        metrics.update(self.custom_metrics)

        return metrics

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

    def set_config(self, config: Dict[str, Any]):
        """Update channel configuration"""
        self.config.update(config)
        # Reinitialize attributes with new config
        self._initialize_from_config()

    def _initialize_from_config(self):
        """Reinitialize attributes from config"""
        # Core metrics
        self.budget = self.config.get('initial_budget', self.budget)
        self.cpi = self.config.get('cpi', self.cpi)
        self.ctr = self.config.get('ctr', self.ctr)
        self.conversion_rate = self.config.get('conversion_rate', self.conversion_rate)
        self.effectiveness_multiplier = self.config.get('effectiveness_multiplier', self.effectiveness_multiplier)
        self.organic_lift_factor = self.config.get('organic_lift_factor', self.organic_lift_factor)

        # Saturation settings
        self.saturation_point = self.config.get('saturation_point', self.saturation_point)
        self.diminishing_returns_rate = self.config.get('diminishing_returns_rate', self.diminishing_returns_rate)

        # Target settings
        self.target_states = self.config.get('target_states', self.target_states)
        self.channel_preference_key = self.config.get('channel_preference_key', self.channel_preference_key)

        # Spend control
        self.daily_spend_limit = self.config.get('daily_spend_limit', self.daily_spend_limit)
        self.daily_spend_rate = self.config.get('daily_spend_rate', self.daily_spend_rate)

        # Custom metrics
        self.custom_metrics = self.config.get('custom_metrics', self.custom_metrics)

    def clone(self, new_name: str, new_config: Dict[str, Any] = None) -> 'UAChannel':
        """Create a copy of this channel with optional new configuration"""
        config_copy = self.config.copy()
        if new_config:
            config_copy.update(new_config)

        # Use the same class type to create new instance
        return self.__class__(self.model, config_copy, new_name, self.channel_type)

    def is_budget_exhausted(self) -> bool:
        """Check if channel budget is exhausted"""
        return self.budget <= 0

    def can_spend_today(self, amount: float) -> bool:
        """Check if channel can spend the specified amount today"""
        if self.is_budget_exhausted():
            return False

        if self.daily_spend_limit:
            return amount <= self.daily_spend_limit

        return True


class PaidSocialChannel(UAChannel):
    """Paid social media advertising (Facebook, Instagram, etc.)"""

    def __init__(self, model, config: Dict[str, Any] = None, name: str = None, channel_type: ChannelType = None):
        # Handle both direct instantiation and cloning
        channel_name = name or 'paid_social'
        channel_type_enum = channel_type or ChannelType.PAID_ACQUISITION

        default_config = {
            'initial_budget': 50000,
            'cpi': 2.5,
            'ctr': 0.03,
            'conversion_rate': 0.08,
            'saturation_point': 8000,
            'effectiveness_multiplier': 1.0,
            'organic_lift_factor': 1.3,
            'daily_spend_rate': 0.1,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'social',
            'description': 'Paid social media advertising campaigns'
        }

        if config:
            default_config.update(config)

        super().__init__(channel_name, channel_type_enum, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize social media specific attributes"""
        self.platform_types = self.config.get('platform_types', ['facebook', 'instagram', 'twitter'])
        self.audience_targeting = self.config.get('audience_targeting', {})
        self.ad_formats = self.config.get('ad_formats', ['image', 'video', 'carousel'])
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 100)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute paid social campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'platform_breakdown': {}
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()

        target_agents = self.get_target_agents(agents)
        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Generate impressions
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        # Randomly assign impressions to agents
        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        # Calculate clicks based on CTR and agent preferences
        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get(self.channel_preference_key, 0.5):
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
        self._update_campaign_metrics(results)

        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


class VideoAdsChannel(UAChannel):
    """Video advertising (TikTok, YouTube, etc.)"""

    def __init__(self, model, config: Dict[str, Any] = None, name: str = None, channel_type: ChannelType = None):
        # Handle both direct instantiation and cloning
        channel_name = name or 'video_ads'
        channel_type_enum = channel_type or ChannelType.PAID_ACQUISITION
        default_config = {
            'initial_budget': 30000,
            'cpi': 3.0,
            'ctr': 0.025,
            'conversion_rate': 0.06,
            'saturation_point': 12000,
            'effectiveness_multiplier': 1.1,
            'organic_lift_factor': 1.4,
            'daily_spend_rate': 0.08,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'video',
            'description': 'Video advertising campaigns on TikTok, YouTube, etc.'
        }

        if config:
            default_config.update(config)

        super().__init__(channel_name, channel_type_enum, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize video ads specific attributes"""
        self.video_length_options = self.config.get('video_length_options', [15, 30, 60])
        self.platform_types = self.config.get('platform_types', ['tiktok', 'youtube', 'instagram_reels'])
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 150)
        self.engagement_bonus = self.config.get('engagement_bonus', 1.2)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute video ads campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'video_views': 0,
            'completion_rate': 0
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Video ads have higher impressions per dollar
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get(self.channel_preference_key, 0.5):
                results['clicks'] += 1

                if random.random() < self.conversion_rate * effectiveness * self.engagement_bonus:
                    results['installs'] += 1

                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps

        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click

        # Estimate video views and completion
        results['video_views'] = int(results['impressions'] * 0.7)  # 70% view rate
        results['completion_rate'] = 0.45  # 45% completion rate

        self._update_campaign_metrics(results)
        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


class SearchAdsChannel(UAChannel):
    """Search advertising (Google Ads, App Store Optimization)"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 20000,
            'cpi': 4.0,
            'ctr': 0.05,
            'conversion_rate': 0.12,
            'saturation_point': 5000,
            'effectiveness_multiplier': 1.2,
            'organic_lift_factor': 1.1,
            'daily_spend_rate': 0.05,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'search',
            'description': 'Search advertising with high intent targeting'
        }

        if config:
            default_config.update(config)

        super().__init__('search_ads', ChannelType.PAID_ACQUISITION, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize search ads specific attributes"""
        self.keyword_categories = self.config.get('keyword_categories', ['brand', 'generic', 'competitor'])
        self.search_engines = self.config.get('search_engines', ['google', 'bing', 'app_store'])
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 50)
        self.intent_multiplier = self.config.get('intent_multiplier', 1.5)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute search ads campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'keyword_performance': {}
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Search has lower impressions but higher intent
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            if random.random() < self.ctr * agent.channel_preference.get(self.channel_preference_key, 0.5):
                results['clicks'] += 1

                # Higher conversion rate due to search intent
                if random.random() < self.conversion_rate * effectiveness * self.intent_multiplier:
                    results['installs'] += 1

                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps

        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click

        self._update_campaign_metrics(results)
        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


class OwnedChannel(UAChannel):
    """Owned channels (push notifications, email, in-game events)"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 0,
            'cpi': 0,
            'ctr': 0.15,
            'retention_boost': 0.1,
            'reactivation_rate': 0.02,
            'target_states': ['installed', 'paying_user'],
            'channel_preference_key': 'owned',
            'description': 'Owned media channels for retention and engagement'
        }

        if config:
            default_config.update(config)

        super().__init__('owned_channels', ChannelType.OWNED_MEDIA, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize owned channel specific attributes"""
        self.channel_types = self.config.get('channel_types', ['push', 'email', 'in_game'])
        self.message_types = self.config.get('message_types', ['promotional', 'transactional', 'engagement'])
        self.reach_limit = self.config.get('reach_limit', None)
        self.iap_boost_factor = self.config.get('iap_boost_factor', 1.2)
        self.retention_boost = self.config.get('retention_boost', 0.1)
        self.reactivation_rate = self.config.get('reactivation_rate', 0.02)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute owned channel campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'reactivations': 0,
            'retention_events': 0,
            'cost': 0,
            'iap_boosts': 0
        }

        # Target installed and paying users
        target_agents = self.get_target_agents(agents)

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
                    agent.iap_probability *= self.iap_boost_factor
                    results['iap_boosts'] += 1

        # Attempt reactivation
        if lapsed_agents and random.random() < self.reactivation_rate:
            reactivate_count = min(int(len(lapsed_agents) * 0.01), 10)
            agents_to_reactivate = random.sample(lapsed_agents, reactivate_count)

            for agent in agents_to_reactivate:
                agent.state = PlayerState.INSTALLED
                results['reactivations'] += 1

        return results


class InfluencerChannel(UAChannel):
    """Paid influencer marketing campaigns"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 25000,
            'cpi': 5.0,
            'ctr': 0.04,
            'conversion_rate': 0.15,
            'saturation_point': 8000,
            'effectiveness_multiplier': 1.5,
            'organic_lift_factor': 2.0,
            'daily_spend_rate': 0.15,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'influencer',
            'description': 'Paid influencer marketing campaigns'
        }

        if config:
            default_config.update(config)

        super().__init__('influencer', ChannelType.PAID_ACQUISITION, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize influencer specific attributes"""
        self.influencer_tiers = self.config.get('influencer_tiers', ['micro', 'macro', 'celebrity'])
        self.content_types = self.config.get('content_types', ['review', 'tutorial', 'unboxing'])
        self.authenticity_bonus = self.config.get('authenticity_bonus', 1.3)
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 80)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute influencer campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'influencer_campaigns': 0,
            'authenticity_score': 0
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Influencer content has moderate impressions but high trust
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            # Higher engagement due to authenticity
            engagement_rate = self.ctr * agent.channel_preference.get(self.channel_preference_key, 0.5) * self.authenticity_bonus
            if random.random() < engagement_rate:
                results['clicks'] += 1

                # Higher conversion due to trusted recommendations
                conversion_rate = self.conversion_rate * effectiveness * self.authenticity_bonus
                if random.random() < conversion_rate:
                    results['installs'] += 1

                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps

        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click

        # Additional metrics
        results['influencer_campaigns'] = max(1, int(daily_budget / 5000))  # Assume $5k per campaign
        results['authenticity_score'] = 0.85  # High authenticity score

        self._update_campaign_metrics(results)
        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


class OOHChannel(UAChannel):
    """Out-of-Home advertising (billboards, transit, etc.)"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 40000,
            'cpi': 6.0,
            'ctr': 0.01,
            'conversion_rate': 0.03,
            'saturation_point': 15000,
            'effectiveness_multiplier': 0.8,
            'organic_lift_factor': 1.5,
            'daily_spend_rate': 0.05,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'ooh',
            'description': 'Out-of-Home advertising campaigns'
        }

        if config:
            default_config.update(config)

        super().__init__('ooh', ChannelType.PAID_ACQUISITION, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize OOH specific attributes"""
        self.ad_formats = self.config.get('ad_formats', ['billboard', 'transit', 'digital'])
        self.geographic_targeting = self.config.get('geographic_targeting', {})
        self.brand_awareness_multiplier = self.config.get('brand_awareness_multiplier', 2.0)
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 200)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute OOH campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'brand_impressions': 0,
            'geographic_coverage': 0
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # OOH has very high impressions but low direct response
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            # Lower CTR due to passive viewing
            if random.random() < self.ctr * agent.channel_preference.get(self.channel_preference_key, 0.5):
                results['clicks'] += 1

                # Lower but steady conversion rate
                if random.random() < self.conversion_rate * effectiveness:
                    results['installs'] += 1

                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps

        cost_per_click = self.cpi * self.conversion_rate
        results['cost'] = results['clicks'] * cost_per_click

        # Additional OOH-specific metrics
        results['brand_impressions'] = impressions  # All impressions count for brand awareness
        results['geographic_coverage'] = min(1.0, daily_budget / 10000)  # Coverage based on spend

        self._update_campaign_metrics(results)
        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


class ProgrammaticDisplayChannel(UAChannel):
    """Programmatic display advertising"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 35000,
            'cpi': 2.0,
            'ctr': 0.02,
            'conversion_rate': 0.04,
            'saturation_point': 20000,
            'effectiveness_multiplier': 0.9,
            'organic_lift_factor': 1.1,
            'daily_spend_rate': 0.12,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'display',
            'description': 'Programmatic display advertising with advanced targeting'
        }

        if config:
            default_config.update(config)

        super().__init__('programmatic_display', ChannelType.PAID_ACQUISITION, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize programmatic display specific attributes"""
        self.targeting_options = self.config.get('targeting_options', ['demographic', 'behavioral', 'contextual'])
        self.ad_sizes = self.config.get('ad_sizes', ['banner', 'native', 'interstitial'])
        self.viewability_rate = self.config.get('viewability_rate', 0.7)
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 300)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute programmatic display campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'viewable_impressions': 0,
            'targeting_efficiency': 0
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Programmatic has very high impressions with advanced targeting
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            # Apply viewability and targeting
            effective_ctr = self.ctr * self.viewability_rate * 1.2  # Targeting bonus
            if random.random() < effective_ctr * agent.channel_preference.get(self.channel_preference_key, 0.5):
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

        # Additional programmatic metrics
        results['viewable_impressions'] = int(impressions * self.viewability_rate)
        results['targeting_efficiency'] = 0.85  # High targeting efficiency

        self._update_campaign_metrics(results)
        return results


class SocialOrganicChannel(UAChannel):
    """Organic social media growth (word of mouth, viral content)"""

    def __init__(self, model, config: Dict[str, Any] = None):
        default_config = {
            'initial_budget': 5000,
            'cpi': 0,
            'ctr': 0.08,
            'conversion_rate': 0.12,
            'saturation_point': 25000,
            'effectiveness_multiplier': 0.6,
            'organic_lift_factor': 3.0,
            'daily_spend_rate': 0.05,
            'target_states': ['unaware', 'aware'],
            'channel_preference_key': 'organic_social',
            'description': 'Organic social media growth and viral content'
        }

        if config:
            default_config.update(config)

        super().__init__('organic_social', ChannelType.EARNED_MEDIA, model, default_config)

    def _initialize_channel_specific_attributes(self):
        """Initialize organic social specific attributes"""
        self.viral_coefficient = self.config.get('viral_coefficient', 1.2)
        self.content_types = self.config.get('content_types', ['viral', 'user_generated', 'community'])
        self.engagement_rate = self.config.get('engagement_rate', 0.15)
        self.impressions_per_dollar = self.config.get('impressions_per_dollar', 500)

    def execute_campaign(self, agents: List[Any]) -> Dict[str, Any]:
        """Execute organic social campaign"""
        results = {
            'impressions': 0,
            'clicks': 0,
            'installs': 0,
            'cost': 0,
            'viral_shares': 0,
            'organic_growth': 0
        }

        if self.is_budget_exhausted() or not self.enabled:
            return results

        daily_budget = self.calculate_daily_budget()
        target_agents = self.get_target_agents(agents)

        if not target_agents:
            return results

        effectiveness = self.calculate_effectiveness(daily_budget)

        # Organic has very high impressions due to sharing
        impressions = int(daily_budget * self.impressions_per_dollar * effectiveness)
        results['impressions'] = impressions

        agents_reached = random.sample(
            target_agents,
            min(impressions, len(target_agents))
        )

        for agent in agents_reached:
            # Higher engagement for organic content
            if random.random() < self.ctr * self.engagement_rate * agent.channel_preference.get(self.channel_preference_key, 0.5):
                results['clicks'] += 1

                # Higher conversion due to trusted sources
                if random.random() < self.conversion_rate * effectiveness * self.viral_coefficient:
                    results['installs'] += 1

                    if agent.state.value == 'unaware':
                        agent.state = PlayerState.AWARE
                    elif agent.state.value == 'aware':
                        agent.state = PlayerState.INSTALLED
                        agent.install_time = self.model.schedule.steps

        # Cost is minimal for organic (mainly content creation)
        results['cost'] = daily_budget * 0.1  # 10% of budget for content creation

        # Organic-specific metrics
        results['viral_shares'] = int(impressions * 0.05)  # 5% share rate
        results['organic_growth'] = int(results['installs'] * 0.3)  # Additional organic installs

        self._update_campaign_metrics(results)
        return results

    def _update_campaign_metrics(self, results: Dict[str, Any]):
        """Update campaign metrics"""
        self.impressions += results['impressions']
        self.clicks += results['clicks']
        self.installs += results['installs']
        self.total_cost += results['cost']
        self.total_spend += results['cost']
        self.budget -= results['cost']


# Channel registry for easy instantiation and management
CHANNEL_REGISTRY = {
    'paid_social': PaidSocialChannel,
    'video_ads': VideoAdsChannel,
    'search_ads': SearchAdsChannel,
    'owned_channels': OwnedChannel,
    'influencer': InfluencerChannel,
    'ooh': OOHChannel,
    'programmatic_display': ProgrammaticDisplayChannel,
    'organic_social': SocialOrganicChannel,
}


def create_channel(channel_type: str, model, config: Dict[str, Any] = None) -> UAChannel:
    """
    Factory function to create channels by type name

    Args:
        channel_type: Name of the channel type
        model: The simulation model
        config: Optional configuration dictionary

    Returns:
        Instance of the specified channel type

    Raises:
        ValueError: If channel type is not supported
    """
    if channel_type not in CHANNEL_REGISTRY:
        raise ValueError(f"Unsupported channel type: {channel_type}. Available types: {list(CHANNEL_REGISTRY.keys())}")

    channel_class = CHANNEL_REGISTRY[channel_type]
    return channel_class(model, config)


def get_available_channels() -> List[str]:
    """Get list of available channel types"""
    return list(CHANNEL_REGISTRY.keys())


def register_channel(channel_name: str, channel_class: Type[UAChannel]):
    """
    Register a new channel type in the registry

    Args:
        channel_name: Name to register the channel under
        channel_class: Channel class to register
    """
    CHANNEL_REGISTRY[channel_name] = channel_class