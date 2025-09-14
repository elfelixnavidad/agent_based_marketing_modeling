import mesa
from enum import Enum
import random
import numpy as np
from typing import Dict, List, Any, Optional
from .user_segmentation import (
    UserSegment, BehaviorPattern, UserPersona, UserSegmentationEngine,
    AcquisitionChannel
)
from .player_persona import PlayerState


class EnhancedPlayerPersona(mesa.Agent):
    """Enhanced player persona with advanced segmentation and behavioral targeting"""

    def __init__(self, unique_id, model, persona_config: Dict[str, Any] = None):
        super().__init__(unique_id, model)
        self.state = PlayerState.UNAWARE

        # Initialize advanced segmentation system
        self.segmentation_engine = UserSegmentationEngine()
        self.persona = self.segmentation_engine.create_persona(persona_config)

        # Initialize tracking variables
        self.install_time = None
        self.total_spend = 0.0
        self.days_active = 0
        self.sessions_played = 0
        self.session_length_history = []
        self.purchase_history = []

        # Behavioral tracking
        self.behavioral_data = {
            "session_length": 0.0,
            "frequency": 0.0,
            "spending": 0.0,
            "social": 0.0,
            "engagement": 0.0,
            "achievement_progress": 0.0,
            "social_interactions": 0
        }

        # Current segmentation
        self.current_segment = UserSegment.NEWBIE
        self.behavioral_patterns = []
        self.targeting_recommendations = {}

        # Enhanced channel preferences based on persona
        self.enhanced_channel_preference = self._calculate_enhanced_channel_preferences()

        # Behavioral targeting attributes
        self.price_sensitivity = self.persona.get_behavioral_score(BehaviorPattern.PRICE_SENSITIVE)
        self.brand_loyalty = self.persona.get_behavioral_score(BehaviorPattern.BRAND_LOYAL)
        self.social_influence_score = self.persona.get_behavioral_score(BehaviorPattern.SOCIAL_INFLUENCER)
        self.achievement_drive = self.persona.get_behavioral_score(BehaviorPattern.ACHIEVEMENT_SEEKER)

        # Adaptation factors
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05

        # Calculate initial segment and patterns
        self._update_segmentation()

    def _calculate_enhanced_channel_preferences(self) -> Dict[str, float]:
        """Calculate enhanced channel preferences based on persona"""
        base_preferences = {
            'social': random.uniform(0.1, 1.0),
            'video': random.uniform(0.1, 1.0),
            'search': random.uniform(0.1, 1.0),
            'organic': random.uniform(0.1, 1.0),
            'influencer': random.uniform(0.1, 1.0),
            'ooh': random.uniform(0.1, 1.0),
            'programmatic': random.uniform(0.1, 1.0),
            'owned': random.uniform(0.1, 1.0)
        }

        # Adjust based on persona attributes
        if self.persona.social_engagement > 0.7:
            base_preferences['social'] *= 1.3
            base_preferences['influencer'] *= 1.2
            base_preferences['organic'] *= 1.1

        if self.persona.technical_proficiency_num() > 0.7:
            base_preferences['search'] *= 1.2
            base_preferences['programmatic'] *= 1.1

        if self.persona.age_group in ["18-24", "25-34"]:
            base_preferences['social'] *= 1.2
            base_preferences['video'] *= 1.2
            base_preferences['influencer'] *= 1.3

        if self.persona.device_type == "mobile":
            base_preferences['social'] *= 1.1
            base_preferences['video'] *= 1.1
            base_preferences['owned'] *= 1.2

        if self.persona.spending_capacity == "high":
            base_preferences['search'] *= 1.1
            base_preferences['programmatic'] *= 1.1

        # Normalize preferences
        max_pref = max(base_preferences.values())
        for key in base_preferences:
            base_preferences[key] = min(base_preferences[key] / max_pref, 1.0)

        return base_preferences

    def _update_segmentation(self):
        """Update user segmentation based on current behavior"""
        # Update behavioral data
        if self.days_active > 0:
            self.behavioral_data["session_length"] = min(self._calculate_avg_session_length() / 30.0, 1.0)
            self.behavioral_data["frequency"] = min(self.sessions_played / max(self.days_active, 1), 1.0)
            self.behavioral_data["spending"] = min(self.total_spend / 100.0, 1.0)
            self.behavioral_data["social"] = min(self.behavioral_data["social_interactions"] / max(self.sessions_played, 1), 1.0)
            self.behavioral_data["engagement"] = min(self._calculate_engagement_score(), 1.0)

        # Update segment
        self.current_segment = self.segmentation_engine.segment_user(self.persona, self.behavioral_data)

        # Update behavioral patterns
        self.behavioral_patterns = self.segmentation_engine.identify_behavioral_patterns(self.persona)

        # Update targeting recommendations
        self.targeting_recommendations = self.segmentation_engine.get_targeting_recommendations(
            self.persona, self.current_segment, self.behavioral_patterns
        )

        # Adapt channel preferences based on behavior
        self._adapt_channel_preferences()

    def _calculate_avg_session_length(self) -> float:
        """Calculate average session length"""
        if not self.session_length_history:
            return 15.0  # Default 15 minutes
        return sum(self.session_length_history) / len(self.session_length_history)

    def _calculate_engagement_score(self) -> float:
        """Calculate overall engagement score"""
        if self.days_active == 0:
            return 0.0

        factors = [
            min(self.sessions_played / self.days_active, 1.0),  # Daily engagement
            min(self.behavioral_data["achievement_progress"], 1.0),  # Achievement engagement
            min(self.behavioral_data["social_interactions"] / max(self.sessions_played, 1), 1.0)  # Social engagement
        ]

        return sum(factors) / len(factors)

    def _adapt_channel_preferences(self):
        """Adapt channel preferences based on behavior and targeting recommendations"""
        if "channel_preferences" in self.targeting_recommendations:
            target_prefs = self.targeting_recommendations["channel_preferences"]

            for channel, target_pref in target_prefs.items():
                current_pref = self.enhanced_channel_preference.get(channel, 0.5)
                adaptation = (target_pref - current_pref) * self.learning_rate

                if abs(adaptation) > self.adaptation_threshold:
                    self.enhanced_channel_preference[channel] = max(0.1, min(1.0, current_pref + adaptation))

    def step(self):
        """Execute one step of the agent's behavior with enhanced segmentation"""
        # Record session data
        if self.state in [PlayerState.INSTALLED, PlayerState.PAYING_USER]:
            self.sessions_played += 1
            session_length = self._generate_session_length()
            self.session_length_history.append(session_length)
            self.behavioral_data["achievement_progress"] += random.uniform(0.1, 0.3)

            # Social interactions based on persona
            if self.persona.social_engagement > random.random():
                self.behavioral_data["social_interactions"] += random.randint(1, 3)

        # Execute base behavior
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

        # Update segmentation periodically
        if self.model.schedule.steps % 7 == 0:  # Update weekly
            self._update_segmentation()

    def _generate_session_length(self) -> float:
        """Generate session length based on persona and segment"""
        base_length = 15.0  # 15 minutes base

        # Adjust based on segment
        segment_multipliers = {
            UserSegment.CASUAL: 0.8,
            UserSegment.CORE: 1.2,
            UserSegment.WHALE: 1.5,
            UserSegment.HIGH_ENGAGEMENT: 1.4,
            UserSegment.LOW_ENGAGEMENT: 0.6,
            UserSegment.SOCIAL_BUTTERFLY: 1.3,
            UserSegment.SOLITARY_PLAYER: 1.1,
        }

        multiplier = segment_multipliers.get(self.current_segment, 1.0)

        # Adjust based on persona
        if self.persona.gaming_frequency == "heavy":
            multiplier *= 1.2
        elif self.persona.gaming_frequency == "light":
            multiplier *= 0.8

        # Add randomness
        session_length = base_length * multiplier * random.uniform(0.5, 1.5)

        return max(5.0, min(120.0, session_length))  # Between 5 and 120 minutes

    def _handle_unaware_state(self):
        """Handle behavior when agent is unaware of the game with segmentation"""
        # Enhanced awareness calculation based on targeting
        awareness_score = self.model.ua_manager.calculate_awareness_impact(self)

        # Apply behavioral targeting
        channel_multiplier = self._get_channel_multiplier_for_awareness()
        awareness_score *= channel_multiplier

        if awareness_score > self._get_install_threshold():
            self.state = PlayerState.AWARE
            self.persona.first_touch_date = self.model.schedule.steps
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.UNAWARE, PlayerState.AWARE
                )

    def _handle_aware_state(self):
        """Handle behavior when agent is aware but hasn't installed with segmentation"""
        install_score = self.model.ua_manager.calculate_install_impact(self)

        # Apply behavioral targeting for install conversion
        behavioral_multiplier = self._get_behavioral_install_multiplier()
        install_score *= behavioral_multiplier

        threshold = self._get_install_threshold()

        if install_score > threshold:
            self.state = PlayerState.INSTALLED
            self.install_time = self.model.schedule.steps
            self.persona.acquisition_channel = self._determine_acquisition_channel()
            self._update_segmentation()  # Initial segmentation after install

            if hasattr(self.model.data_collector, 'record_install'):
                self.model.data_collector.record_install(self.unique_id)
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.AWARE, PlayerState.INSTALLED
                )

    def _handle_installed_state(self):
        """Handle behavior when agent is an active player with segmentation"""
        self.days_active += 1

        # Enhanced churn calculation based on segmentation
        churn_probability = self._calculate_segmented_churn_probability()

        if random.random() < churn_probability:
            self.state = PlayerState.LAPSED
            self.persona.last_active_date = self.model.schedule.steps
            if hasattr(self.model.data_collector, 'record_churn'):
                self.model.data_collector.record_churn(self.unique_id)
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.INSTALLED, PlayerState.LAPSED
                )
            return

        # Enhanced IAP calculation based on segmentation
        iap_probability = self._calculate_segmented_iap_probability()
        if random.random() < iap_probability:
            self._make_segmented_purchase()

        # Enhanced organic influence based on social behavior
        if self.persona.social_engagement > 0.6:
            self._spread_awareness_enhanced()

        # Engage with owned channels with personalized targeting
        self.model.ua_manager.handle_owned_channel_interaction(self)

    def _handle_paying_user_state(self):
        """Handle behavior when agent is a paying user with segmentation"""
        self._handle_installed_state()

        # Enhanced retention for paying users based on segment
        retention_bonus = {
            UserSegment.WHALE: 0.5,
            UserSegment.CORE: 0.4,
            UserSegment.HIGH_ENGAGEMENT: 0.3,
        }.get(self.current_segment, 0.2)

        self.churn_propensity *= (1.0 - retention_bonus)

    def _handle_lapsed_state(self):
        """Handle behavior when agent has churned with segmentation"""
        # Enhanced reactivation based on segmentation
        reactivation_chance = self._calculate_segmented_reactivation_chance()

        if random.random() < reactivation_chance:
            self.state = PlayerState.INSTALLED
            if hasattr(self.model.data_collector, 'record_reactivation'):
                self.model.data_collector.record_reactivation(self.unique_id)

    def _get_channel_multiplier_for_awareness(self) -> float:
        """Get channel effectiveness multiplier based on persona"""
        if BehaviorPattern.TREND_FOLLOWER in self.behavioral_patterns:
            return 1.3
        elif BehaviorPattern.SOCIAL_INFLUENCER in self.behavioral_patterns:
            return 1.4
        elif BehaviorPattern.EARLY_ADOPTER in self.behavioral_patterns:
            return 1.2
        return 1.0

    def _get_behavioral_install_multiplier(self) -> float:
        """Get behavioral multiplier for install conversion"""
        multiplier = 1.0

        if BehaviorPattern.BRAND_LOYAL in self.behavioral_patterns:
            multiplier *= 1.2

        if self.persona.novelty_seeking > 0.7:
            multiplier *= 1.1

        if self.current_segment == UserSegment.HIGH_ENGAGEMENT:
            multiplier *= 1.2

        return multiplier

    def _get_install_threshold(self) -> float:
        """Get install threshold based on persona and segment"""
        base_threshold = 0.3 + (1.0 - self.persona.risk_tolerance) * 0.3

        # Adjust based on segment
        segment_adjustments = {
            UserSegment.HIGH_ENGAGEMENT: -0.1,
            UserSegment.CORE: -0.05,
            UserSegment.LOW_ENGAGEMENT: 0.1,
        }

        return max(0.1, min(0.9, base_threshold + segment_adjustments.get(self.current_segment, 0.0)))

    def _calculate_segmented_churn_probability(self) -> float:
        """Calculate churn probability based on segmentation"""
        base_churn = self.persona.churn_propensity

        # Segment-based adjustments
        segment_multipliers = {
            UserSegment.WHALE: 0.5,
            UserSegment.CORE: 0.7,
            UserSegment.HIGH_ENGAGEMENT: 0.6,
            UserSegment.CASUAL: 1.0,
            UserSegment.LOW_ENGAGEMENT: 1.5,
        }

        churn_prob = base_churn * segment_multipliers.get(self.current_segment, 1.0)

        # Behavioral adjustments
        if BehaviorPattern.BRAND_LOYAL in self.behavioral_patterns:
            churn_prob *= 0.7

        if self.behavioral_data["engagement"] > 0.7:
            churn_prob *= 0.6

        # Day-based churn curve
        if self.days_active < 7:
            churn_prob *= 2.0
        elif self.days_active < 30:
            churn_prob *= 1.3

        return max(0.001, min(0.1, churn_prob))

    def _calculate_segmented_iap_probability(self) -> float:
        """Calculate IAP probability based on segmentation"""
        base_iap = self.persona.iap_probability

        # Segment-based adjustments
        segment_multipliers = {
            UserSegment.WHALE: 3.0,
            UserSegment.CORE: 2.0,
            UserSegment.HIGH_ENGAGEMENT: 1.8,
            UserSegment.CASUAL: 1.0,
            UserSegment.LOW_ENGAGEMENT: 0.5,
        }

        iap_prob = base_iap * segment_multipliers.get(self.current_segment, 1.0)

        # Behavioral adjustments
        if BehaviorPattern.PRICE_SENSITIVE in self.behavioral_patterns:
            iap_prob *= 0.7

        if self.persona.spending_capacity_num() > 0.7:
            iap_prob *= 1.3

        return max(0.001, min(0.5, iap_prob))

    def _calculate_segmented_reactivation_chance(self) -> float:
        """Calculate reactivation chance based on segmentation"""
        base_chance = 0.001  # 0.1% base chance

        # Segment-based adjustments
        segment_multipliers = {
            UserSegment.CORE: 5.0,
            UserSegment.WHALE: 10.0,
            UserSegment.HIGH_ENGAGEMENT: 3.0,
            UserSegment.RETURNING: 2.0,
        }

        reactivation_chance = base_chance * segment_multipliers.get(self.current_segment, 1.0)

        # Behavioral adjustments
        if BehaviorPattern.BRAND_LOYAL in self.behavioral_patterns:
            reactivation_chance *= 2.0

        if self.persona.social_engagement > 0.7:
            reactivation_chance *= 1.5

        return min(0.05, reactivation_chance)  # Max 5% chance

    def _make_segmented_purchase(self):
        """Make a purchase with segmentation-based amounts"""
        if self.state == PlayerState.INSTALLED:
            self.state = PlayerState.PAYING_USER
            if hasattr(self.model.data_collector, 'record_state_change'):
                self.model.data_collector.record_state_change(
                    self.unique_id, PlayerState.INSTALLED, PlayerState.PAYING_USER
                )

        # Segmented purchase amounts
        if self.current_segment == UserSegment.WHALE:
            purchase_amount = np.random.pareto(0.8) * 10.0 + 4.99  # $5-50+
        elif self.current_segment == UserSegment.CORE:
            purchase_amount = np.random.pareto(1.0) * 5.0 + 1.99  # $2-20
        elif BehaviorPattern.PRICE_SENSITIVE in self.behavioral_patterns:
            purchase_amount = np.random.pareto(1.5) * 2.0 + 0.99  # $1-5
        else:
            purchase_amount = np.random.pareto(1.2) * 3.0 + 1.49  # $1.5-15

        purchase_amount = min(purchase_amount, 99.99)

        self.total_spend += purchase_amount
        self.purchase_history.append({
            "amount": purchase_amount,
            "date": self.model.schedule.steps,
            "segment": self.current_segment.value
        })

        if hasattr(self.model.data_collector, 'record_purchase'):
            self.model.data_collector.record_purchase(self.unique_id, purchase_amount)

    def _spread_awareness_enhanced(self):
        """Enhanced organic awareness spreading based on social behavior"""
        if self.persona.k_factor > 0 and self.persona.social_engagement > 0.5:
            unaware_agents = [
                agent for agent in self.model.schedule.agents
                if agent.state == PlayerState.UNAWARE and agent.unique_id != self.unique_id
            ]

            if unaware_agents:
                # Enhanced influence based on social behavior
                influence_multiplier = self.persona.social_engagement * self.social_influence_score
                num_to_influence = min(int(self.persona.k_factor * 10 * influence_multiplier), len(unaware_agents))

                if num_to_influence > 0:
                    agents_to_influence = random.sample(unaware_agents, num_to_influence)

                    for agent in agents_to_influence:
                        # Higher success rate for social influencers
                        success_rate = 0.3 * influence_multiplier
                        if random.random() < success_rate:
                            agent.state = PlayerState.AWARE
                            if hasattr(self.model.data_collector, 'record_organic_conversion'):
                                self.model.data_collector.record_organic_conversion(
                                    self.unique_id, agent.unique_id
                                )

    def _determine_acquisition_channel(self) -> AcquisitionChannel:
        """Determine acquisition channel based on persona and behavior"""
        # This would normally be determined by the UA channel that converted the user
        # For simulation purposes, we'll infer based on persona

        if self.persona.social_engagement > 0.7:
            return AcquisitionChannel.PAID_SOCIAL
        elif self.persona.technical_proficiency_num() > 0.7:
            return AcquisitionChannel.ORGANIC_SEARCH
        elif self.persona.novelty_seeking > 0.7:
            return AcquisitionChannel.INFLUENCER
        else:
            return AcquisitionChannel.ORGANIC_SEARCH

    def get_enhanced_ltv(self) -> Dict[str, float]:
        """Get enhanced LTV calculations with segmentation"""
        base_ltv = self.total_spend

        # Predictive LTV based on segment and behavior
        segment_ltv_multipliers = {
            UserSegment.WHALE: 10.0,
            UserSegment.CORE: 3.0,
            UserSegment.HIGH_ENGAGEMENT: 2.5,
            UserSegment.CASUAL: 1.0,
            UserSegment.LOW_ENGAGEMENT: 0.5,
        }

        predicted_ltv = base_ltv * segment_ltv_multipliers.get(self.current_segment, 1.0)

        # Adjust based on behavioral patterns
        if BehaviorPattern.BRAND_LOYAL in self.behavioral_patterns:
            predicted_ltv *= 1.5

        if BehaviorPattern.PRICE_SENSITIVE in self.behavioral_patterns:
            predicted_ltv *= 0.7

        return {
            "current_ltv": base_ltv,
            "predicted_ltv": predicted_ltv,
            "segment_multiplier": segment_ltv_multipliers.get(self.current_segment, 1.0)
        }

    def get_segmentation_insights(self) -> Dict[str, Any]:
        """Get comprehensive segmentation insights"""
        return {
            "current_segment": self.current_segment.value,
            "behavioral_patterns": [p.value for p in self.behavioral_patterns],
            "persona_summary": {
                "age_group": self.persona.age_group,
                "gaming_frequency": self.persona.gaming_frequency,
                "spending_capacity": self.persona.spending_capacity,
                "social_engagement": self.persona.social_engagement,
                "technical_proficiency": self.persona.technical_proficiency,
            },
            "behavioral_data": self.behavioral_data,
            "targeting_recommendations": self.targeting_recommendations,
            "channel_preferences": self.enhanced_channel_preference,
            "enhanced_ltv": self.get_enhanced_ltv()
        }