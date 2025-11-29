"""
Property-based tests for Monad M: Recursive Meta-Prompting.

Uses Hypothesis to verify categorical laws with 1000+ random examples:
1. Left Identity Law: unit(a) >>= f = f(a)
2. Right Identity Law: m >>= unit = m
3. Associativity Law: (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

References:
- L5 Meta-Prompt: "Monad M for recursive improvement with quality join"
- Moggi (1991) - Computational monads for programming language semantics
- Zhang et al. (arXiv:2311.11482) - 100% on Game of 24 with meta-prompting
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Callable
from datetime import datetime

from meta_prompting_engine.categorical.types import Prompt, QualityScore
from meta_prompting_engine.categorical.monad import (
    Monad,
    MonadPrompt,
    create_recursive_meta_monad,
    kleisli_compose
)


# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client that returns predictable outputs for testing."""

    def __init__(self, base_quality: float = 0.75):
        self.base_quality = base_quality
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        """Return mock completion based on prompt."""
        self.call_count += 1

        # Simulate different outputs based on prompt complexity
        if "step-by-step" in prompt.lower():
            return f"Detailed solution with reasoning (call {self.call_count})"
        elif "approach" in prompt.lower():
            return f"Multi-approach solution (call {self.call_count})"
        else:
            return f"Basic solution (call {self.call_count})"


# Hypothesis strategies for generating random prompts
@st.composite
def prompt_strategy(draw):
    """Generate random Prompt objects for property-based testing."""
    templates = [
        "Solve: {task}",
        "You are a {role}. Task: {task}",
        "Using {strategy}, solve: {task}",
        "Step-by-step solution for: {task}",
        "Provide multiple approaches to: {task}"
    ]

    template = draw(st.sampled_from(templates))

    variables = {}
    if "{task}" in template:
        variables["task"] = draw(st.text(min_size=10, max_size=50))
    if "{role}" in template:
        variables["role"] = draw(st.sampled_from([
            "python programmer", "system architect", "data scientist"
        ]))
    if "{strategy}" in template:
        variables["strategy"] = draw(st.sampled_from([
            "divide-and-conquer", "dynamic programming", "greedy approach"
        ]))

    meta_level = draw(st.integers(min_value=0, max_value=3))

    return Prompt(
        template=template,
        variables=variables,
        context={
            'test': True,
            'generated': datetime.now().isoformat()
        },
        meta_level=meta_level
    )


@st.composite
def monadic_function_strategy(draw):
    """Generate random monadic functions Prompt → M(Prompt)."""
    mock_llm = MockLLMClient(base_quality=draw(st.floats(min_value=0.6, max_value=0.9)))
    monad = create_recursive_meta_monad(mock_llm)

    transformations = [
        lambda p: monad.unit(Prompt(
            template=f"{p.template} (enhanced)",
            variables=p.variables,
            context={**p.context, 'enhanced': True},
            meta_level=p.meta_level
        )),
        lambda p: monad.unit(Prompt(
            template=f"Refine: {p.template}",
            variables=p.variables,
            context={**p.context, 'refined': True},
            meta_level=p.meta_level
        )),
        lambda p: monad.unit(Prompt(
            template=f"Optimize: {p.template}",
            variables=p.variables,
            context={**p.context, 'optimized': True},
            meta_level=p.meta_level
        ))
    ]

    return draw(st.sampled_from(transformations))


class TestMonadLaws:
    """
    Property-based tests for Monad categorical laws.

    Monad M: P → M(P) must satisfy:
    1. unit(a) >>= f = f(a) (left identity)
    2. m >>= unit = m (right identity)
    3. (m >>= f) >>= g = m >>= (λx. f(x) >>= g) (associativity)
    """

    @pytest.fixture
    def monad(self) -> Monad:
        """Create monad for testing."""
        mock_llm = MockLLMClient(base_quality=0.80)
        return create_recursive_meta_monad(mock_llm, quality_threshold=0.90)

    @settings(max_examples=1000, deadline=None)
    @given(
        prompt=prompt_strategy(),
        quality=st.floats(min_value=0.6, max_value=0.9)
    )
    def test_monad_left_identity_law(
        self,
        monad: Monad,
        prompt: Prompt,
        quality: float
    ):
        """
        Test Monad Left Identity Law: unit(a) >>= f = f(a)

        Wrapping a value with unit and then binding should equal
        just applying the function directly.

        Mathematical Definition:
            η(a) >>= f = f(a)

        Args:
            monad: The monad M
            prompt: Random prompt
            quality: Random quality for mock LLM
        """
        # Update mock LLM quality
        monad_llm = MockLLMClient(base_quality=quality)
        test_monad = create_recursive_meta_monad(monad_llm)

        # Monadic function: Prompt → M(Prompt)
        def f(p: Prompt) -> MonadPrompt:
            return test_monad.unit(Prompt(
                template=f"Enhanced: {p.template}",
                variables=p.variables,
                context={**p.context, 'enhanced': True},
                meta_level=p.meta_level
            ))

        # Left side: unit(prompt) >>= f
        left_side = test_monad.bind(test_monad.unit(prompt), f)

        # Right side: f(prompt)
        right_side = f(prompt)

        # Verify equality (structural, quality may differ slightly)
        assert test_monad._monadic_prompts_equal(left_side, right_side), \
            f"Left identity violated: unit(p) >>= f ≠ f(p) for prompt: {prompt.template[:50]}"

    @settings(max_examples=1000, deadline=None)
    @given(
        prompt=prompt_strategy(),
        quality=st.floats(min_value=0.6, max_value=0.9)
    )
    def test_monad_right_identity_law(
        self,
        monad: Monad,
        prompt: Prompt,
        quality: float
    ):
        """
        Test Monad Right Identity Law: m >>= unit = m

        Binding a monadic value with unit should return
        the original monadic value.

        Mathematical Definition:
            m >>= η = m

        Args:
            monad: The monad M
            prompt: Random prompt
            quality: Random quality for mock LLM
        """
        # Update mock LLM quality
        monad_llm = MockLLMClient(base_quality=quality)
        test_monad = create_recursive_meta_monad(monad_llm)

        # Create monadic prompt
        m = test_monad.unit(prompt)

        # Left side: m >>= unit
        left_side = test_monad.bind(m, test_monad.unit)

        # Right side: m
        right_side = m

        # Verify equality
        assert test_monad._monadic_prompts_equal(left_side, right_side), \
            f"Right identity violated: m >>= unit ≠ m for prompt: {prompt.template[:50]}"

    @settings(max_examples=500, deadline=None)  # Reduced for performance
    @given(
        prompt=prompt_strategy(),
        quality=st.floats(min_value=0.6, max_value=0.9)
    )
    def test_monad_associativity_law(
        self,
        monad: Monad,
        prompt: Prompt,
        quality: float
    ):
        """
        Test Monad Associativity Law: (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

        The order of binding operations should not matter.

        Mathematical Definition:
            (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

        Args:
            monad: The monad M
            prompt: Random prompt
            quality: Random quality for mock LLM
        """
        # Update mock LLM quality
        monad_llm = MockLLMClient(base_quality=quality)
        test_monad = create_recursive_meta_monad(monad_llm)

        # Create monadic prompt
        m = test_monad.unit(prompt)

        # Define two monadic functions
        def f(p: Prompt) -> MonadPrompt:
            return test_monad.unit(Prompt(
                template=f"Step 1: {p.template}",
                variables=p.variables,
                context={**p.context, 'step1': True},
                meta_level=p.meta_level
            ))

        def g(p: Prompt) -> MonadPrompt:
            return test_monad.unit(Prompt(
                template=f"Step 2: {p.template}",
                variables=p.variables,
                context={**p.context, 'step2': True},
                meta_level=p.meta_level
            ))

        # Left side: (m >>= f) >>= g
        left_side = test_monad.bind(test_monad.bind(m, f), g)

        # Right side: m >>= (λx. f(x) >>= g)
        right_side = test_monad.bind(m, lambda x: test_monad.bind(f(x), g))

        # Verify equality
        assert test_monad._monadic_prompts_equal(left_side, right_side), \
            f"Associativity violated: (m >>= f) >>= g ≠ m >>= (λx. f(x) >>= g) for prompt: {prompt.template[:50]}"

    @settings(max_examples=100, deadline=None)
    @given(
        prompt=prompt_strategy(),
        quality=st.floats(min_value=0.6, max_value=0.9)
    )
    def test_runtime_law_verification(
        self,
        monad: Monad,
        prompt: Prompt,
        quality: float
    ):
        """
        Test runtime verification methods.

        Verifies that the monad's built-in law verification
        methods correctly validate categorical properties.
        """
        # Update mock LLM quality
        monad_llm = MockLLMClient(base_quality=quality)
        test_monad = create_recursive_meta_monad(monad_llm)

        # Define monadic functions
        def f(p: Prompt) -> MonadPrompt:
            return test_monad.unit(Prompt(
                template=f"Enhanced: {p.template}",
                variables=p.variables,
                context={**p.context, 'enhanced': True},
                meta_level=p.meta_level
            ))

        def g(p: Prompt) -> MonadPrompt:
            return test_monad.unit(Prompt(
                template=f"Refined: {p.template}",
                variables=p.variables,
                context={**p.context, 'refined': True},
                meta_level=p.meta_level
            ))

        # Verify left identity using runtime method
        assert test_monad.verify_left_identity(prompt, f), \
            f"Runtime left identity verification failed for: {prompt.template[:50]}"

        # Verify right identity using runtime method
        m = test_monad.unit(prompt)
        assert test_monad.verify_right_identity(m), \
            f"Runtime right identity verification failed for: {prompt.template[:50]}"

        # Verify associativity using runtime method
        assert test_monad.verify_associativity(m, f, g), \
            f"Runtime associativity verification failed for: {prompt.template[:50]}"


class TestMonadImplementation:
    """
    Additional tests for monad implementation details.

    Not categorical laws, but important for correctness.
    """

    @pytest.fixture
    def monad(self) -> Monad:
        """Create monad for testing."""
        mock_llm = MockLLMClient(base_quality=0.80)
        return create_recursive_meta_monad(mock_llm, quality_threshold=0.90)

    @settings(max_examples=100, deadline=None)
    @given(prompt=prompt_strategy())
    def test_quality_assessment(self, monad: Monad, prompt: Prompt):
        """Test that monad correctly assesses output quality."""
        mp = monad.unit(prompt)

        # Verify quality was assessed
        assert mp.quality is not None, "Monad must assess quality"
        assert isinstance(mp.quality, QualityScore), "Quality must be QualityScore"
        assert 0.0 <= mp.quality.value <= 1.0, "Quality must be in [0.0, 1.0]"

        # Verify quality has components
        assert mp.quality.components, "Quality must have component breakdown"

    @settings(max_examples=100, deadline=None)
    @given(prompt=prompt_strategy())
    def test_meta_level_tracking(self, monad: Monad, prompt: Prompt):
        """Test that monad correctly tracks meta-level depth."""
        # Initial prompt has meta_level 0
        mp = monad.unit(prompt)
        assert mp.meta_level == 0, "Initial prompt should have meta_level 0"

        # Bind increases meta_level
        def f(p: Prompt) -> MonadPrompt:
            return monad.unit(Prompt(
                template=f"Enhanced: {p.template}",
                variables=p.variables,
                context=p.context,
                meta_level=p.meta_level
            ))

        mp2 = monad.bind(mp, f)
        assert mp2.meta_level == 1, "Bind should increase meta_level"

    @settings(max_examples=100, deadline=None)
    @given(prompt=prompt_strategy())
    def test_history_tracking(self, monad: Monad, prompt: Prompt):
        """Test that monad maintains improvement history."""
        mp1 = monad.unit(prompt)

        # Initially empty history
        assert len(mp1.history) == 0, "Initial prompt should have empty history"

        # Bind adds to history
        def f(p: Prompt) -> MonadPrompt:
            return monad.unit(Prompt(
                template=f"Enhanced: {p.template}",
                variables=p.variables,
                context=p.context,
                meta_level=p.meta_level
            ))

        mp2 = monad.bind(mp1, f)
        assert len(mp2.history) >= 1, "Bind should add to history"

    @settings(max_examples=100, deadline=None)
    @given(
        prompt=prompt_strategy(),
        quality1=st.floats(min_value=0.6, max_value=0.9),
        quality2=st.floats(min_value=0.6, max_value=0.9)
    )
    def test_quality_tensor_product(
        self,
        monad: Monad,
        prompt: Prompt,
        quality1: float,
        quality2: float
    ):
        """Test that bind correctly computes quality tensor product."""
        q1 = QualityScore(value=quality1)
        q2 = QualityScore(value=quality2)

        # Tensor product should be minimum
        result = q1.tensor_product(q2)
        assert result.value == min(quality1, quality2), \
            "Tensor product should be minimum of qualities"


class TestKleisliComposition:
    """
    Tests for Kleisli composition (>=>).

    The categorical composition of monadic functions.
    """

    @pytest.fixture
    def monad(self) -> Monad:
        """Create monad for testing."""
        mock_llm = MockLLMClient(base_quality=0.80)
        return create_recursive_meta_monad(mock_llm)

    @settings(max_examples=100, deadline=None)
    @given(prompt=prompt_strategy())
    def test_kleisli_composition(self, monad: Monad, prompt: Prompt):
        """Test Kleisli composition: (f >=> g)(a) = f(a) >>= g"""

        def f(p: Prompt) -> MonadPrompt:
            return monad.unit(Prompt(
                template=f"Step 1: {p.template}",
                variables=p.variables,
                context={**p.context, 'step1': True},
                meta_level=p.meta_level
            ))

        def g(p: Prompt) -> MonadPrompt:
            return monad.unit(Prompt(
                template=f"Step 2: {p.template}",
                variables=p.variables,
                context={**p.context, 'step2': True},
                meta_level=p.meta_level
            ))

        # Compose using Kleisli
        f_then_g = kleisli_compose(monad, f, g)

        # Apply composition
        composed_result = f_then_g(prompt)

        # Apply manually: f(prompt) >>= g
        manual_result = monad.bind(f(prompt), g)

        # Results should be equal
        assert monad._monadic_prompts_equal(composed_result, manual_result), \
            "Kleisli composition should equal manual bind"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
