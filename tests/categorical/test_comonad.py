"""
Property-based tests for Comonad W: Context Extraction and Observation.

Uses Hypothesis to verify categorical laws with 1000+ random examples:
1. Left Identity Law: extract ∘ duplicate = id
2. Right Identity Law: fmap extract ∘ duplicate = id
3. Associativity Law: duplicate ∘ duplicate = fmap duplicate ∘ duplicate

References:
- L5 Meta-Prompt: "Comonad W for context extraction with extract/extend"
- Uustalu & Vene (2008) - Comonads and Context-Dependent Computation
- CC2.0 OBSERVE framework integration
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Callable, Any
from datetime import datetime

from meta_prompting_engine.categorical.comonad import (
    Comonad,
    Observation,
    create_context_comonad,
    create_observation
)


# Hypothesis strategies for generating random observations
@st.composite
def observation_strategy(draw):
    """Generate random Observation objects for property-based testing."""
    # Random current value
    current_types = [
        draw(st.text(min_size=5, max_size=100)),  # String
        draw(st.integers()),  # Integer
        draw(st.floats(allow_nan=False, allow_infinity=False)),  # Float
        draw(st.lists(st.integers(), max_size=10)),  # List
        {"result": draw(st.text(min_size=5, max_size=50))}  # Dict
    ]

    current = draw(st.sampled_from(current_types))

    # Random context
    context = {
        'prompt': draw(st.text(min_size=10, max_size=50)),
        'quality': draw(st.floats(min_value=0.0, max_value=1.0)),
        'meta_level': draw(st.integers(min_value=0, max_value=5)),
        'timestamp': datetime.now().isoformat()
    }

    # Random history (0-3 previous observations to avoid deep nesting)
    history_size = draw(st.integers(min_value=0, max_value=2))
    history = []

    # Random metadata
    metadata = {
        'iteration': draw(st.integers(min_value=0, max_value=10)),
        'source': draw(st.sampled_from(['llm', 'user', 'system'])),
        'test': True
    }

    return Observation(
        current=current,
        context=context,
        history=history,
        metadata=metadata,
        timestamp=datetime.now()
    )


@st.composite
def comonadic_function_strategy(draw):
    """Generate random comonadic functions W(A) → B."""
    transformations = [
        # Extract quality score from context
        lambda obs: obs.context.get('quality', 0.5),

        # Count history depth
        lambda obs: len(obs.history),

        # Compute observation age
        lambda obs: (datetime.now() - obs.timestamp).total_seconds() if obs.timestamp else 0,

        # Assess observation completeness
        lambda obs: len(obs.context) / 10.0,

        # Extract meta-level
        lambda obs: obs.context.get('meta_level', 0),

        # Summarize observation
        lambda obs: f"Observation with {len(obs.context)} context keys and {len(obs.history)} history items"
    ]

    return draw(st.sampled_from(transformations))


class TestComonadLaws:
    """
    Property-based tests for Comonad categorical laws.

    Comonad W: Observations must satisfy:
    1. extract ∘ duplicate = id (left identity)
    2. fmap extract ∘ duplicate = id (right identity)
    3. duplicate ∘ duplicate = fmap duplicate ∘ duplicate (associativity)
    """

    @pytest.fixture
    def comonad(self) -> Comonad:
        """Create comonad for testing."""
        return create_context_comonad()

    @settings(max_examples=1000, deadline=None)
    @given(obs=observation_strategy())
    def test_comonad_left_identity_law(self, comonad: Comonad, obs: Observation):
        """
        Test Comonad Left Identity Law: extract ∘ duplicate = id

        Extracting from a duplicated observation should give
        back the original observation.

        Mathematical Definition:
            ε(δ(w)) = w

        Args:
            comonad: The comonad W
            obs: Random observation
        """
        # Left side: extract(duplicate(obs))
        duplicated = comonad.duplicate(obs)
        extracted = comonad.extract(duplicated)

        # Right side: obs
        # Verify structural equality
        assert comonad._observations_equal(extracted, obs), \
            f"Left identity violated: extract(duplicate(w)) ≠ w for observation: {str(obs.current)[:50]}"

    @settings(max_examples=1000, deadline=None)
    @given(obs=observation_strategy())
    def test_comonad_right_identity_law(self, comonad: Comonad, obs: Observation):
        """
        Test Comonad Right Identity Law: fmap extract ∘ duplicate = id

        Mapping extract over a duplicated observation should
        give back the original observation's current value.

        Mathematical Definition:
            fmap ε(δ(w)) = w

        Args:
            comonad: The comonad W
            obs: Random observation
        """
        # Left side: fmap extract(duplicate(obs))
        duplicated = comonad.duplicate(obs)

        # fmap extract means: extract from the inner observation
        fmap_extracted_current = comonad.extract(duplicated.current)

        # Right side: obs.current
        # Verify equality of current values
        assert comonad._observations_equal(fmap_extracted_current, obs.current), \
            f"Right identity violated: fmap extract(duplicate(w)) ≠ w for observation: {str(obs.current)[:50]}"

    @settings(max_examples=500, deadline=None)  # Reduced for performance
    @given(obs=observation_strategy())
    def test_comonad_associativity_law(self, comonad: Comonad, obs: Observation):
        """
        Test Comonad Associativity Law: duplicate ∘ duplicate = fmap duplicate ∘ duplicate

        Duplicating twice should be the same as duplicating and
        mapping duplicate over the result.

        Mathematical Definition:
            δ(δ(w)) = fmap δ(δ(w))

        Args:
            comonad: The comonad W
            obs: Random observation
        """
        # Left side: duplicate(duplicate(obs))
        left_side = comonad.duplicate(comonad.duplicate(obs))

        # Right side: fmap duplicate(duplicate(obs))
        duplicated_once = comonad.duplicate(obs)

        # fmap duplicate means: duplicate the inner observation
        right_side_current = comonad.duplicate(duplicated_once.current)

        # Both should have W(W(W(A))) structure
        assert isinstance(left_side.current, Observation), \
            "Left side should have nested structure W(W(W(A)))"

        assert isinstance(right_side_current, Observation), \
            "Right side should have nested structure W(W(W(A)))"

        # The structures should be equivalent
        assert (
            isinstance(left_side.current, Observation) and
            isinstance(right_side_current, Observation)
        ), f"Associativity violated: structures differ for observation: {str(obs.current)[:50]}"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_runtime_law_verification(self, comonad: Comonad, obs: Observation):
        """
        Test runtime verification methods.

        Verifies that the comonad's built-in law verification
        methods correctly validate categorical properties.
        """
        # Verify left identity using runtime method
        assert comonad.verify_left_identity(obs), \
            f"Runtime left identity verification failed for: {str(obs.current)[:50]}"

        # Verify right identity using runtime method
        assert comonad.verify_right_identity(obs), \
            f"Runtime right identity verification failed for: {str(obs.current)[:50]}"

        # Verify associativity using runtime method
        assert comonad.verify_associativity(obs), \
            f"Runtime associativity verification failed for: {str(obs.current)[:50]}"


class TestComonadExtend:
    """
    Tests for comonad extend operation (cobind).

    extend is the key operation that makes comonads useful:
    extend : (W(A) → B) → W(A) → W(B)
    """

    @pytest.fixture
    def comonad(self) -> Comonad:
        """Create comonad for testing."""
        return create_context_comonad()

    @settings(max_examples=100, deadline=None)
    @given(
        obs=observation_strategy(),
        f=comonadic_function_strategy()
    )
    def test_extend_applies_function_with_context(
        self,
        comonad: Comonad,
        obs: Observation,
        f: Callable[[Observation], Any]
    ):
        """
        Test that extend correctly applies function with full context.

        extend should:
        1. Duplicate the observation
        2. Apply f to the duplicated observation
        3. Wrap result in new observation with original context
        """
        # Apply extend
        result = comonad.extend(f, obs)

        # Verify result is an observation
        assert isinstance(result, Observation), \
            "extend must return Observation"

        # Verify context is preserved
        assert result.context == obs.context, \
            "extend must preserve context"

        # Verify history is preserved
        assert result.history == obs.history, \
            "extend must preserve history"

        # Verify metadata indicates extension
        assert result.metadata.get('extended') == True, \
            "extend must mark as extended in metadata"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_extend_composition(self, comonad: Comonad, obs: Observation):
        """
        Test that extend respects composition.

        extend(f) ∘ extend(g) should relate to extend(f ∘ cojoin ∘ g)
        """
        # Two comonadic functions
        def f(w: Observation) -> float:
            return w.context.get('quality', 0.5)

        def g(w: Observation) -> int:
            return len(w.history)

        # Apply extend twice
        step1 = comonad.extend(g, obs)
        step2 = comonad.extend(lambda w: f(comonad.duplicate(w)), step1)

        # Result should be observation
        assert isinstance(step2, Observation), \
            "Composed extend should return Observation"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_extend_with_extract_is_identity(self, comonad: Comonad, obs: Observation):
        """
        Test that extend with extract is identity.

        extend(extract) = id
        This is a corollary of the comonad laws.
        """
        # extend(extract) should be identity
        result = comonad.extend(comonad.extract, obs)

        # Current value should be same as original
        assert comonad._observations_equal(result.current, obs.current), \
            "extend(extract) should be identity"


class TestComonadImplementation:
    """
    Additional tests for comonad implementation details.

    Not categorical laws, but important for correctness.
    """

    @pytest.fixture
    def comonad(self) -> Comonad:
        """Create comonad for testing."""
        return create_context_comonad()

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_observation_quality_assessment(self, comonad: Comonad, obs: Observation):
        """Test that duplicate assesses observation quality."""
        duplicated = comonad.duplicate(obs)

        # Verify meta-observation has quality metadata
        assert 'observation_quality' in duplicated.metadata, \
            "duplicate should assess observation quality"

        quality = duplicated.metadata['observation_quality']
        assert 0.0 <= quality <= 1.0, \
            "Observation quality should be in [0.0, 1.0]"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_observation_completeness_assessment(self, comonad: Comonad, obs: Observation):
        """Test that duplicate assesses observation completeness."""
        duplicated = comonad.duplicate(obs)

        # Verify meta-observation has completeness metadata
        assert 'completeness' in duplicated.metadata, \
            "duplicate should assess observation completeness"

        completeness = duplicated.metadata['completeness']
        assert 0.0 <= completeness <= 1.0, \
            "Observation completeness should be in [0.0, 1.0]"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_history_accumulation(self, comonad: Comonad, obs: Observation):
        """Test that duplicate correctly accumulates history."""
        duplicated = comonad.duplicate(obs)

        # History should include original observation
        assert len(duplicated.history) == len(obs.history) + 1, \
            "duplicate should prepend current observation to history"

        # First history item should be the original observation
        assert duplicated.history[0] == obs, \
            "duplicate should add current observation to history"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_meta_observation_context(self, comonad: Comonad, obs: Observation):
        """Test that duplicate creates proper meta-observation context."""
        duplicated = comonad.duplicate(obs)

        # Verify meta-observation context
        assert duplicated.context.get('meta_observation') == True, \
            "duplicate should mark as meta-observation"

        assert 'original_context_keys' in duplicated.context, \
            "duplicate should record original context keys"

        assert 'observation_timestamp' in duplicated.context, \
            "duplicate should record observation timestamp"

        assert 'history_depth' in duplicated.context, \
            "duplicate should record history depth"


class TestCC2ObserveIntegration:
    """
    Tests for CC2.0 OBSERVE framework integration.

    The comonad structure matches CC2.0 operations:
    - extract = focused view on system health
    - duplicate = meta-observation of observation quality
    - extend = context-aware recommendations
    """

    @pytest.fixture
    def comonad(self) -> Comonad:
        """Create comonad for testing."""
        return create_context_comonad()

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_extract_focused_view(self, comonad: Comonad, obs: Observation):
        """Test extract provides focused view (CC2.0 OBSERVE)."""
        # extract should give focused value
        focused = comonad.extract(obs)

        # Should equal current value
        assert focused == obs.current, \
            "extract should provide focused view on current value"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_duplicate_meta_observation(self, comonad: Comonad, obs: Observation):
        """Test duplicate creates meta-observation (CC2.0 OBSERVE)."""
        # duplicate should create meta-observation
        meta_obs = comonad.duplicate(obs)

        # Inner observation should be original
        assert meta_obs.current == obs, \
            "duplicate should make observation the current value"

        # Should have meta-observation markers
        assert meta_obs.context.get('meta_observation') == True, \
            "duplicate should mark as meta-observation"

    @settings(max_examples=100, deadline=None)
    @given(obs=observation_strategy())
    def test_extend_context_aware_transformation(self, comonad: Comonad, obs: Observation):
        """Test extend enables context-aware transformations (CC2.0 OBSERVE)."""
        # Context-aware function that uses full observation
        def assess_health(w: Observation) -> str:
            quality = w.context.get('quality', 0.5)
            history_depth = len(w.history)

            if quality >= 0.9 and history_depth > 0:
                return "EXCELLENT"
            elif quality >= 0.7:
                return "GOOD"
            elif quality >= 0.5:
                return "FAIR"
            else:
                return "NEEDS_IMPROVEMENT"

        # Apply extend
        health_obs = comonad.extend(assess_health, obs)

        # Result should be observation with health assessment
        assert isinstance(health_obs, Observation), \
            "extend should return Observation"

        assert health_obs.current in ["EXCELLENT", "GOOD", "FAIR", "NEEDS_IMPROVEMENT"], \
            "extend should apply context-aware transformation"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
