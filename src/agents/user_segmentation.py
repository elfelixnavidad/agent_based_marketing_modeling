import random
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


class UserSegment(Enum):
    """User segmentation categories based on behavior and demographics"""
    CASUAL = "casual"
    CORE = "core"
    WHALE = "whale"
    NEWBIE = "newbie"
    RETURNING = "returning"
    CHURNED = "churned"
    HIGH_ENGAGEMENT = "high_engagement"
    LOW_ENGAGEMENT = "low_engagement"
    SOCIAL_BUTTERFLY = "social_butterfly"
    SOLITARY_PLAYER = "solitary_player"
    COMPETITIVE = "competitive"
    CASUAL_SOCIAL = "casual_social"


class BehaviorPattern(Enum):
    """Behavioral patterns for targeting"""
    PRICE_SENSITIVE = "price_sensitive"
    BRAND_LOYAL = "brand_loyal"
    TREND_FOLLOWER = "trend_follower"
    EARLY_ADOPTER = "early_adopter"
    LATE_ADOPTER = "late_adopter"
    SOCIAL_INFLUENCER = "social_influencer"
    ACHIEVEMENT_SEEKER = "achievement_seeker"
    EXPLORER = "explorer"
    COLLECTOR = "collector"


class AcquisitionChannel(Enum):
    """Primary acquisition channels for segmentation"""
    ORGANIC_SEARCH = "organic_search"
    PAID_SOCIAL = "paid_social"
    INFLUENCER = "influencer"
    REFERRAL = "referral"
    APP_STORE = "app_store"
    WEB_DIRECT = "web_direct"
    EMAIL_MARKETING = "email_marketing"
    CONTENT_MARKETING = "content_marketing"


@dataclass
class UserPersona:
    """Detailed user persona for advanced segmentation"""
    age_group: str = "25-34"
    gender: str = "unknown"
    location: str = "us"
    device_type: str = "mobile"
    os_type: str = "android"
    income_bracket: str = "middle"
    education_level: str = "college"
    occupation: str = "professional"
    gaming_frequency: str = "moderate"
    spending_capacity: str = "medium"
    technical_proficiency: str = "average"

    # Behavioral attributes
    risk_tolerance: float = 0.5
    novelty_seeking: float = 0.5
    social_engagement: float = 0.5
    competition_drive: float = 0.5
    achievement_motivation: float = 0.5
    exploration_tendency: float = 0.5
    collection_drive: float = 0.5

    # Gaming preferences
    preferred_game_genres: List[str] = field(default_factory=lambda: random.sample(["casual", "puzzle", "strategy", "action", "rpg", "simulation", "sports", "adventure"], 3))
    session_length_preference: str = "medium"
    play_time_preference: List[str] = field(default_factory=lambda: ["evening"])
    social_play_preference: float = 0.5

    # Marketing preferences
    ad_tolerance: float = 0.5
    email_preference: float = 0.5
    push_notification_preference: float = 0.5
    social_sharing_likelihood: float = 0.3

    # Lifecycle attributes
    acquisition_channel: AcquisitionChannel = AcquisitionChannel.ORGANIC_SEARCH
    acquisition_cost: float = 0.0
    first_touch_date: Optional[int] = None
    last_active_date: Optional[int] = None

    # Behavioral attributes (for integration with player persona)
    churn_propensity: float = 0.03
    iap_probability: float = 0.1

    def get_behavioral_score(self, pattern: BehaviorPattern) -> float:
        """Calculate behavioral score for targeting"""
        score_map = {
            BehaviorPattern.PRICE_SENSITIVE: (1.0 - self.spending_capacity_num()) * self.risk_tolerance,
            BehaviorPattern.BRAND_LOYAL: (1.0 - self.novelty_seeking) * self.social_engagement,
            BehaviorPattern.TREND_FOLLOWER: self.novelty_seeking * self.social_engagement,
            BehaviorPattern.EARLY_ADOPTER: self.novelty_seeking * self.technical_proficiency_num(),
            BehaviorPattern.LATE_ADOPTER: (1.0 - self.novelty_seeking) * (1.0 - self.risk_tolerance),
            BehaviorPattern.SOCIAL_INFLUENCER: self.social_engagement * self.social_sharing_likelihood,
            BehaviorPattern.ACHIEVEMENT_SEEKER: self.achievement_motivation * self.competition_drive,
            BehaviorPattern.EXPLORER: self.exploration_tendency * self.novelty_seeking,
            BehaviorPattern.COLLECTOR: self.collection_drive * self.achievement_motivation,
        }
        return score_map.get(pattern, 0.5)

    def spending_capacity_num(self) -> float:
        """Convert spending capacity to numerical value"""
        capacity_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "premium": 1.0}
        return capacity_map.get(self.spending_capacity, 0.5)

    def technical_proficiency_num(self) -> float:
        """Convert technical proficiency to numerical value"""
        proficiency_map = {"beginner": 0.2, "average": 0.5, "advanced": 0.8, "expert": 1.0}
        return proficiency_map.get(self.technical_proficiency, 0.5)


class UserSegmentationEngine:
    """Advanced user segmentation and behavioral targeting engine"""

    def __init__(self):
        self.segment_weights = {
            UserSegment.CASUAL: {"session_length": 0.3, "frequency": 0.4, "spending": 0.2, "social": 0.1},
            UserSegment.CORE: {"session_length": 0.6, "frequency": 0.7, "spending": 0.5, "social": 0.3},
            UserSegment.WHALE: {"session_length": 0.8, "frequency": 0.9, "spending": 1.0, "social": 0.4},
            UserSegment.HIGH_ENGAGEMENT: {"session_length": 0.7, "frequency": 0.8, "spending": 0.3, "social": 0.6},
            UserSegment.LOW_ENGAGEMENT: {"session_length": 0.2, "frequency": 0.3, "spending": 0.1, "social": 0.2},
        }

        self.behavioral_thresholds = {
            BehaviorPattern.PRICE_SENSITIVE: 0.7,
            BehaviorPattern.BRAND_LOYAL: 0.6,
            BehaviorPattern.TREND_FOLLOWER: 0.6,
            BehaviorPattern.EARLY_ADOPTER: 0.7,
            BehaviorPattern.SOCIAL_INFLUENCER: 0.8,
            BehaviorPattern.ACHIEVEMENT_SEEKER: 0.7,
        }

    def create_persona(self, config: Dict[str, Any] = None) -> UserPersona:
        """Create a user persona with optional configuration"""
        persona = UserPersona()

        # Generate random attributes first
        persona.age_group = random.choice(["18-24", "25-34", "35-44", "45-54", "55+"])
        persona.gender = random.choice(["male", "female", "non_binary", "unknown"])
        persona.location = random.choice(["us", "eu", "asia", "latam", "other"])
        persona.device_type = random.choice(["mobile", "tablet", "desktop"])

        # Override with config if provided
        if config:
            for key, value in config.items():
                if hasattr(persona, key):
                    setattr(persona, key, value)

        # Generate behavioral attributes
        persona.risk_tolerance = random.uniform(0.1, 0.9)
        persona.novelty_seeking = random.uniform(0.1, 0.9)
        persona.social_engagement = random.uniform(0.1, 0.9)
        persona.competition_drive = random.uniform(0.1, 0.9)
        persona.achievement_motivation = random.uniform(0.1, 0.9)
        persona.exploration_tendency = random.uniform(0.1, 0.9)
        persona.collection_drive = random.uniform(0.1, 0.9)

        # Set gaming preferences
        genre_pool = ["casual", "puzzle", "strategy", "action", "rpg", "simulation", "sports", "adventure"]
        num_genres = random.randint(1, 3)
        persona.preferred_game_genres = random.sample(genre_pool, k=min(num_genres, len(genre_pool)))

        return persona

    def segment_user(self, persona: UserPersona, behavioral_data: Dict[str, float]) -> UserSegment:
        """Segment user based on persona and behavioral data"""
        scores = {}

        # Calculate segment scores
        for segment, weights in self.segment_weights.items():
            score = 0
            for metric, weight in weights.items():
                if metric in behavioral_data:
                    score += behavioral_data[metric] * weight
            scores[segment] = score

        # Apply additional persona-based rules
        if persona.spending_capacity == "high" and behavioral_data.get("spending", 0) > 0.8:
            scores[UserSegment.WHALE] *= 1.5

        if persona.social_engagement > 0.7 and behavioral_data.get("social", 0) > 0.6:
            scores[UserSegment.SOCIAL_BUTTERFLY] = (scores.get(UserSegment.SOCIAL_BUTTERFLY, 0) +
                                                    behavioral_data.get("social", 0) * persona.social_engagement)

        if persona.social_engagement < 0.3 and behavioral_data.get("social", 0) < 0.3:
            scores[UserSegment.SOLITARY_PLAYER] = (scores.get(UserSegment.SOLITARY_PLAYER, 0) +
                                                   (1.0 - behavioral_data.get("social", 0)) * (1.0 - persona.social_engagement))

        # Special handling for low activity users (newbies/casual)
        total_activity = sum(behavioral_data.values())
        if total_activity < 1.0:  # Low overall activity
            if persona.spending_capacity in ["low", "medium"]:
                scores[UserSegment.NEWBIE] = max(scores.get(UserSegment.NEWBIE, 0), 0.8)
            else:
                scores[UserSegment.CASUAL] = max(scores.get(UserSegment.CASUAL, 0), 0.7)

        # Penalize whale segment for low-spending users
        if persona.spending_capacity in ["low", "medium"] and behavioral_data.get("spending", 0) < 0.3:
            scores[UserSegment.WHALE] *= 0.1

        # Return segment with highest score
        return max(scores, key=scores.get)

    def identify_behavioral_patterns(self, persona: UserPersona) -> List[BehaviorPattern]:
        """Identify behavioral patterns for targeting"""
        patterns = []

        for pattern in BehaviorPattern:
            score = persona.get_behavioral_score(pattern)
            threshold = self.behavioral_thresholds.get(pattern, 0.5)

            if score >= threshold:
                patterns.append(pattern)

        return patterns

    def get_targeting_recommendations(self, persona: UserPersona, segment: UserSegment,
                                    patterns: List[BehaviorPattern]) -> Dict[str, Any]:
        """Generate targeting recommendations for user"""
        recommendations = {
            "segment": segment.value,
            "behavioral_patterns": [p.value for p in patterns],
            "channel_preferences": {},
            "message_strategy": {},
            "offer_strategy": {},
            "timing_strategy": {}
        }

        # Channel preferences based on segment and behavior
        if UserSegment.WHALE in [segment]:
            recommendations["channel_preferences"] = {
                "email": 0.9,
                "push": 0.8,
                "in_app": 0.9,
                "social": 0.6
            }
        elif UserSegment.CASUAL in [segment]:
            recommendations["channel_preferences"] = {
                "social": 0.8,
                "push": 0.6,
                "email": 0.4,
                "in_app": 0.7
            }
        else:
            recommendations["channel_preferences"] = {
                "push": 0.7,
                "email": 0.6,
                "social": 0.6,
                "in_app": 0.8
            }

        # Adjust based on behavioral patterns
        if BehaviorPattern.PRICE_SENSITIVE in patterns:
            recommendations["offer_strategy"] = {
                "discount_sensitivity": "high",
                "preferred_offers": ["percentage_off", "free_trial", "bonus_currency"],
                "price_range": "low_to_medium"
            }

        if BehaviorPattern.SOCIAL_INFLUENCER in patterns:
            recommendations["message_strategy"] = {
                "tone": "social_proof",
                "content_type": "user_generated",
                "sharing_incentive": "high"
            }

        if BehaviorPattern.ACHIEVEMENT_SEEKER in patterns:
            recommendations["message_strategy"] = {
                "tone": "achievement_focused",
                "content_type": "progress_milestones",
                "gamification": "high"
            }

        # Timing based on persona
        if "evening" in persona.play_time_preference:
            recommendations["timing_strategy"] = {
                "best_send_times": ["18:00", "20:00", "22:00"],
                "frequency": "moderate"
            }

        return recommendations

    def calculate_segment_affinity(self, persona: UserPersona, target_segment: UserSegment) -> float:
        """Calculate affinity between user persona and target segment"""
        affinity = 0.0

        # Base affinity on spending capacity
        if target_segment == UserSegment.WHALE:
            affinity += persona.spending_capacity_num() * 0.5
        elif target_segment == UserSegment.CORE:
            affinity += min(persona.spending_capacity_num() * 0.7, 0.7)
        elif target_segment == UserSegment.CASUAL:
            affinity += (1.0 - persona.spending_capacity_num()) * 0.5

        # Add behavioral alignment
        if target_segment in [UserSegment.HIGH_ENGAGEMENT, UserSegment.SOCIAL_BUTTERFLY]:
            affinity += persona.social_engagement * 0.3

        if target_segment == UserSegment.SOLITARY_PLAYER:
            affinity += (1.0 - persona.social_engagement) * 0.3

        return min(affinity, 1.0)

    def create_behavioral_lookalike(self, source_persona: UserPersona,
                                   target_persona: UserPersona) -> Dict[str, float]:
        """Create behavioral lookalike scoring between personas"""
        similarity_scores = {}

        # Behavioral attribute similarity
        behavioral_attrs = [
            'risk_tolerance', 'novelty_seeking', 'social_engagement', 'competition_drive',
            'achievement_motivation', 'exploration_tendency', 'collection_drive'
        ]

        for attr in behavioral_attrs:
            source_val = getattr(source_persona, attr)
            target_val = getattr(target_persona, attr)
            similarity = 1.0 - abs(source_val - target_val)
            similarity_scores[attr] = similarity

        # Gaming preference similarity
        source_genres = set(source_persona.preferred_game_genres)
        target_genres = set(target_persona.preferred_game_genres)
        genre_similarity = len(source_genres.intersection(target_genres)) / max(len(source_genres.union(target_genres)), 1)
        similarity_scores["gaming_preference"] = genre_similarity

        # Overall similarity
        overall_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        similarity_scores["overall_similarity"] = overall_similarity

        return similarity_scores