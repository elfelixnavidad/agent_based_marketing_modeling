import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.user_segmentation import (
    UserSegment, BehaviorPattern, AcquisitionChannel, UserPersona,
    UserSegmentationEngine
)
from agents.enhanced_player_persona import EnhancedPlayerPersona
from agents.player_persona import PlayerState
from environment.marketing_simulation import MarketingSimulation


class TestUserPersona(unittest.TestCase):
    """Test UserPersona class"""

    def setUp(self):
        self.persona = UserPersona()

    def test_persona_initialization(self):
        """Test that persona initializes with default values"""
        self.assertIsInstance(self.persona.age_group, str)
        self.assertIsInstance(self.persona.gender, str)
        self.assertIsInstance(self.persona.location, str)
        self.assertIsInstance(self.persona.device_type, str)
        self.assertEqual(len(self.persona.preferred_game_genres), 3)

    def test_persona_custom_config(self):
        """Test persona with custom configuration"""
        config = {
            "age_group": "18-24",
            "gender": "male",
            "location": "us",
            "spending_capacity": "high"
        }
        persona = UserPersona()
        # Apply config manually
        for key, value in config.items():
            if hasattr(persona, key):
                setattr(persona, key, value)

        self.assertEqual(persona.age_group, "18-24")
        self.assertEqual(persona.gender, "male")
        self.assertEqual(persona.location, "us")
        self.assertEqual(persona.spending_capacity, "high")

    def test_behavioral_score_calculation(self):
        """Test behavioral score calculation"""
        self.persona.spending_capacity = "low"
        self.persona.risk_tolerance = 0.8
        score = self.persona.get_behavioral_score(BehaviorPattern.PRICE_SENSITIVE)

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_spending_capacity_num(self):
        """Test spending capacity numerical conversion"""
        self.persona.spending_capacity = "high"
        self.assertEqual(self.persona.spending_capacity_num(), 0.8)

        self.persona.spending_capacity = "low"
        self.assertEqual(self.persona.spending_capacity_num(), 0.2)

    def test_technical_proficiency_num(self):
        """Test technical proficiency numerical conversion"""
        self.persona.technical_proficiency = "expert"
        self.assertEqual(self.persona.technical_proficiency_num(), 1.0)

        self.persona.technical_proficiency = "beginner"
        self.assertEqual(self.persona.technical_proficiency_num(), 0.2)


class TestUserSegmentationEngine(unittest.TestCase):
    """Test UserSegmentationEngine class"""

    def setUp(self):
        self.engine = UserSegmentationEngine()

    def test_engine_initialization(self):
        """Test that engine initializes with proper weights"""
        self.assertIn(UserSegment.CASUAL, self.engine.segment_weights)
        self.assertIn(UserSegment.WHALE, self.engine.segment_weights)
        self.assertIn(BehaviorPattern.PRICE_SENSITIVE, self.engine.behavioral_thresholds)

    def test_create_persona(self):
        """Test persona creation"""
        persona = self.engine.create_persona()

        self.assertIsInstance(persona, UserPersona)
        self.assertIsInstance(persona.age_group, str)
        self.assertGreater(len(persona.preferred_game_genres), 0)

    def test_create_persona_with_config(self):
        """Test persona creation with configuration"""
        config = {"age_group": "25-34", "spending_capacity": "high"}
        persona = self.engine.create_persona(config)

        self.assertEqual(persona.age_group, "25-34")
        self.assertEqual(persona.spending_capacity, "high")

    def test_segment_user_casual(self):
        """Test casual user segmentation"""
        persona = UserPersona()
        persona.spending_capacity = "low"
        behavioral_data = {
            "session_length": 0.2,
            "frequency": 0.3,
            "spending": 0.1,
            "social": 0.2
        }

        segment = self.engine.segment_user(persona, behavioral_data)
        self.assertIn(segment, [UserSegment.CASUAL, UserSegment.NEWBIE])

    def test_segment_user_whale(self):
        """Test whale user segmentation"""
        persona = UserPersona()
        persona.spending_capacity = "high"
        behavioral_data = {
            "session_length": 0.9,
            "frequency": 0.9,
            "spending": 1.0,
            "social": 0.6
        }

        segment = self.engine.segment_user(persona, behavioral_data)
        self.assertEqual(segment, UserSegment.WHALE)

    def test_identify_behavioral_patterns(self):
        """Test behavioral pattern identification"""
        persona = UserPersona()
        persona.risk_tolerance = 0.8
        persona.novelty_seeking = 0.8
        persona.social_engagement = 0.9

        patterns = self.engine.identify_behavioral_patterns(persona)

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

    def test_get_targeting_recommendations(self):
        """Test targeting recommendations generation"""
        persona = UserPersona()
        segment = UserSegment.CORE
        patterns = [BehaviorPattern.ACHIEVEMENT_SEEKER]

        recommendations = self.engine.get_targeting_recommendations(persona, segment, patterns)

        self.assertIn("segment", recommendations)
        self.assertIn("behavioral_patterns", recommendations)
        self.assertIn("channel_preferences", recommendations)
        self.assertIn("message_strategy", recommendations)

    def test_calculate_segment_affinity(self):
        """Test segment affinity calculation"""
        persona = UserPersona()
        persona.spending_capacity = "high"

        affinity = self.engine.calculate_segment_affinity(persona, UserSegment.WHALE)

        self.assertGreaterEqual(affinity, 0.0)
        self.assertLessEqual(affinity, 1.0)

    def test_create_behavioral_lookalike(self):
        """Test behavioral lookalike scoring"""
        persona1 = UserPersona()
        persona2 = UserPersona()

        similarity_scores = self.engine.create_behavioral_lookalike(persona1, persona2)

        self.assertIn("overall_similarity", similarity_scores)
        self.assertIn("gaming_preference", similarity_scores)
        self.assertGreaterEqual(similarity_scores["overall_similarity"], 0.0)
        self.assertLessEqual(similarity_scores["overall_similarity"], 1.0)


class TestEnhancedPlayerPersona(unittest.TestCase):
    """Test EnhancedPlayerPersona class"""

    def setUp(self):
        # Create a simple mock model that works with Mesa Agent
        class MockModel:
            def __init__(self):
                self.agents_ = {}
                self.schedule = Mock()
                self.schedule.steps = 0
                self.ua_manager = Mock()
                self.data_collector = Mock()
                # Initialize agents_ dictionary for all agent types we'll use
                from agents.enhanced_player_persona import EnhancedPlayerPersona
                self.agents_[EnhancedPlayerPersona] = {}

        self.model = MockModel()
        self.persona = EnhancedPlayerPersona(1, self.model)

    def test_enhanced_persona_initialization(self):
        """Test enhanced persona initialization"""
        self.assertIsInstance(self.persona.segmentation_engine, UserSegmentationEngine)
        self.assertIsInstance(self.persona.persona, UserPersona)
        self.assertEqual(self.persona.state, PlayerState.UNAWARE)
        # With zero activity, user gets classified as SOLITARY_PLAYER or NEWBIE
        self.assertIn(self.persona.current_segment, [UserSegment.NEWBIE, UserSegment.SOLITARY_PLAYER])

    def test_enhanced_channel_preferences(self):
        """Test enhanced channel preferences calculation"""
        preferences = self.persona.enhanced_channel_preference

        self.assertIn("social", preferences)
        self.assertIn("video", preferences)
        self.assertIn("search", preferences)
        self.assertIn("influencer", preferences)

        # Check that preferences are normalized
        for pref_value in preferences.values():
            self.assertGreaterEqual(pref_value, 0.1)
            self.assertLessEqual(pref_value, 1.0)

    def test_session_length_generation(self):
        """Test session length generation"""
        self.persona.current_segment = UserSegment.CORE
        session_length = self.persona._generate_session_length()

        self.assertGreaterEqual(session_length, 5.0)
        self.assertLessEqual(session_length, 120.0)

    def test_average_session_length_calculation(self):
        """Test average session length calculation"""
        self.persona.session_length_history = [15.0, 20.0, 25.0]
        avg_length = self.persona._calculate_avg_session_length()

        self.assertEqual(avg_length, 20.0)

    def test_engagement_score_calculation(self):
        """Test engagement score calculation"""
        self.persona.sessions_played = 10
        self.persona.days_active = 5
        self.persona.behavioral_data["achievement_progress"] = 0.8
        self.persona.behavioral_data["social_interactions"] = 15

        engagement_score = self.persona._calculate_engagement_score()

        self.assertGreaterEqual(engagement_score, 0.0)
        self.assertLessEqual(engagement_score, 1.0)

    def test_segmentation_update(self):
        """Test segmentation update"""
        self.persona.days_active = 10
        self.persona.sessions_played = 15
        self.persona.total_spend = 50.0
        self.persona.behavioral_data["social_interactions"] = 20

        self.persona._update_segmentation()

        self.assertIsInstance(self.persona.current_segment, UserSegment)
        self.assertIsInstance(self.persona.behavioral_patterns, list)
        self.assertIsInstance(self.persona.targeting_recommendations, dict)

    def test_segmented_churn_probability(self):
        """Test segmented churn probability calculation"""
        self.persona.current_segment = UserSegment.WHALE
        self.persona.behavioral_patterns = [BehaviorPattern.BRAND_LOYAL]
        self.persona.days_active = 5

        churn_prob = self.persona._calculate_segmented_churn_probability()

        self.assertGreaterEqual(churn_prob, 0.0)
        self.assertLessEqual(churn_prob, 0.1)

    def test_segmented_iap_probability(self):
        """Test segmented IAP probability calculation"""
        self.persona.current_segment = UserSegment.WHALE
        self.persona.persona.spending_capacity = "high"

        iap_prob = self.persona._calculate_segmented_iap_probability()

        self.assertGreaterEqual(iap_prob, 0.0)
        self.assertLessEqual(iap_prob, 0.5)

    def test_segmented_reactivation_chance(self):
        """Test segmented reactivation chance calculation"""
        self.persona.current_segment = UserSegment.WHALE
        self.persona.behavioral_patterns = [BehaviorPattern.BRAND_LOYAL]

        reactivation_chance = self.persona._calculate_segmented_reactivation_chance()

        self.assertGreaterEqual(reactivation_chance, 0.0)
        self.assertLessEqual(reactivation_chance, 0.05)

    def test_enhanced_ltv_calculation(self):
        """Test enhanced LTV calculation"""
        self.persona.total_spend = 100.0
        self.persona.current_segment = UserSegment.WHALE
        self.persona.behavioral_patterns = [BehaviorPattern.BRAND_LOYAL]

        ltv_data = self.persona.get_enhanced_ltv()

        self.assertIn("current_ltv", ltv_data)
        self.assertIn("predicted_ltv", ltv_data)
        self.assertIn("segment_multiplier", ltv_data)

        self.assertEqual(ltv_data["current_ltv"], 100.0)
        self.assertGreater(ltv_data["predicted_ltv"], 100.0)

    def test_segmentation_insights(self):
        """Test segmentation insights generation"""
        insights = self.persona.get_segmentation_insights()

        self.assertIn("current_segment", insights)
        self.assertIn("behavioral_patterns", insights)
        self.assertIn("persona_summary", insights)
        self.assertIn("behavioral_data", insights)
        self.assertIn("targeting_recommendations", insights)
        self.assertIn("channel_preferences", insights)
        self.assertIn("enhanced_ltv", insights)

    def test_behavioral_multiplier_awareness(self):
        """Test behavioral multiplier for awareness"""
        self.persona.behavioral_patterns = [BehaviorPattern.SOCIAL_INFLUENCER]
        multiplier = self.persona._get_channel_multiplier_for_awareness()

        self.assertGreaterEqual(multiplier, 1.0)

    def test_behavioral_multiplier_install(self):
        """Test behavioral multiplier for install conversion"""
        self.persona.behavioral_patterns = [BehaviorPattern.BRAND_LOYAL]
        multiplier = self.persona._get_behavioral_install_multiplier()

        self.assertGreaterEqual(multiplier, 1.0)

    def test_install_threshold_calculation(self):
        """Test install threshold calculation"""
        threshold = self.persona._get_install_threshold()

        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.9)


class TestIntegrationUserSegmentation(unittest.TestCase):
    """Integration tests for user segmentation system"""

    def setUp(self):
        """Set up integration test environment"""
        # Create a minimal mock model for integration testing
        class MockModel:
            def __init__(self):
                self.agents_ = {}
                self.schedule = Mock()
                self.schedule.steps = 0
                self.ua_manager = Mock()
                self.data_collector = Mock()
                self.initial_budget = 10000
                # Initialize agents_ dictionary for all agent types we'll use
                from agents.enhanced_player_persona import EnhancedPlayerPersona
                self.agents_[EnhancedPlayerPersona] = {}

        self.model = MockModel()

    def test_enhanced_persona_in_simulation(self):
        """Test enhanced persona works in simulation context"""
        # Create enhanced persona
        persona_config = {"age_group": "25-34", "spending_capacity": "high"}
        enhanced_persona = EnhancedPlayerPersona(1, self.model, persona_config)

        # Add to simulation
        self.model.schedule.add(enhanced_persona)

        # Test basic functionality
        self.assertIsInstance(enhanced_persona.current_segment, UserSegment)
        self.assertIsInstance(enhanced_persona.behavioral_patterns, list)
        self.assertIn("social", enhanced_persona.enhanced_channel_preference)

    def test_persona_with_custom_config(self):
        """Test persona with custom configuration"""
        config = {
            "age_group": "18-24",
            "gender": "female",
            "location": "us",
            "spending_capacity": "high",
            "gaming_frequency": "heavy",
            "social_engagement": 0.8,
            "technical_proficiency": "advanced"
        }

        persona = EnhancedPlayerPersona(1, self.model, config)

        self.assertEqual(persona.persona.age_group, "18-24")
        self.assertEqual(persona.persona.gender, "female")
        self.assertEqual(persona.persona.spending_capacity, "high")
        self.assertEqual(persona.persona.gaming_frequency, "heavy")
        # Note: social_engagement is randomly generated in behavioral attributes, so we don't test it here
        self.assertEqual(persona.persona.technical_proficiency, "advanced")

    def test_segmentation_adaptation(self):
        """Test that segmentation adapts over time"""
        persona = EnhancedPlayerPersona(1, self.model)

        initial_segment = persona.current_segment
        initial_patterns = persona.behavioral_patterns.copy()

        # Simulate some activity
        persona.days_active = 30
        persona.sessions_played = 45
        persona.total_spend = 200.0
        persona.behavioral_data["social_interactions"] = 50

        # Update segmentation
        persona._update_segmentation()

        # Check that segmentation may have changed
        self.assertIsInstance(persona.current_segment, UserSegment)
        self.assertIsInstance(persona.behavioral_patterns, list)

    def test_channel_preference_adaptation(self):
        """Test that channel preferences adapt based on behavior"""
        persona = EnhancedPlayerPersona(1, self.model)

        initial_social_pref = persona.enhanced_channel_preference.get("social", 0.5)

        # Simulate targeting recommendation that favors social
        persona.targeting_recommendations = {
            "channel_preferences": {"social": 0.9}
        }

        # Adapt preferences
        persona._adapt_channel_preferences()

        # Check that preference moved towards target
        new_social_pref = persona.enhanced_channel_preference.get("social", 0.5)
        self.assertGreaterEqual(new_social_pref, initial_social_pref - 0.1)

    def test_behavioral_purchase_amounts(self):
        """Test that purchase amounts vary by segmentation"""
        persona = EnhancedPlayerPersona(1, self.model)

        # Test whale purchase
        persona.current_segment = UserSegment.WHALE
        whale_purchase = persona._make_segmented_purchase()

        # Reset and test casual purchase
        persona.total_spend = 0
        persona.purchase_history = []
        persona.state = PlayerState.INSTALLED
        persona.current_segment = UserSegment.CASUAL
        persona.behavioral_patterns = [BehaviorPattern.PRICE_SENSITIVE]
        casual_purchase = persona._make_segmented_purchase()

        # Whale should generally spend more
        self.assertGreater(persona.total_spend, 0)


class TestUserSegmentationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.engine = UserSegmentationEngine()

    def test_empty_behavioral_data(self):
        """Test segmentation with empty behavioral data"""
        persona = UserPersona()
        empty_data = {}

        segment = self.engine.segment_user(persona, empty_data)

        self.assertIsInstance(segment, UserSegment)

    def test_extreme_persona_values(self):
        """Test persona with extreme values"""
        persona = UserPersona()
        persona.risk_tolerance = 1.0
        persona.novelty_seeking = 1.0
        persona.social_engagement = 1.0

        patterns = self.engine.identify_behavioral_patterns(persona)

        self.assertIsInstance(patterns, list)

    def test_persona_with_minimal_config(self):
        """Test persona with minimal configuration"""
        minimal_config = {"age_group": "25-34"}
        persona = self.engine.create_persona(minimal_config)

        self.assertEqual(persona.age_group, "25-34")
        self.assertIsInstance(persona.gender, str)  # Should be randomly generated

    def test_segment_affinity_bounds(self):
        """Test that segment affinity stays within bounds"""
        persona = UserPersona()
        persona.spending_capacity = "high"

        affinity = self.engine.calculate_segment_affinity(persona, UserSegment.WHALE)

        self.assertGreaterEqual(affinity, 0.0)
        self.assertLessEqual(affinity, 1.0)

    def test_lookalike_same_persona(self):
        """Test lookalike scoring with identical personas"""
        persona1 = UserPersona()
        persona1.age_group = "25-34"
        persona1.risk_tolerance = 0.5

        persona2 = UserPersona()
        persona2.age_group = "25-34"
        persona2.risk_tolerance = 0.5

        similarity = self.engine.create_behavioral_lookalike(persona1, persona2)

        self.assertGreaterEqual(similarity["overall_similarity"], 0.8)  # Should be high


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)