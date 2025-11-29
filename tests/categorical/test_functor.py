"""
Property-based tests for Functor F: Tasks → Prompts.

Uses Hypothesis to verify categorical laws with 1000+ random examples:
1. Identity Law: F(id) = id
2. Composition Law: F(g ∘ f) = F(g) ∘ F(f)

References:
- L5 Meta-Prompt: "Functor F: Tasks → Prompts with verified identity/composition laws"
- Mac Lane (1971) - Categories for the Working Mathematician
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Callable

from meta_prompting_engine.categorical.types import Task, Prompt, ComplexityAnalysis
from meta_prompting_engine.categorical.functor import (
    create_task_to_prompt_functor,
    Functor
)
from meta_prompting_engine.categorical.complexity import analyze_complexity
from meta_prompting_engine.categorical.strategy import select_strategy


# Hypothesis strategies for generating random tasks
@st.composite
def task_strategy(draw):
    """Generate random Task objects for property-based testing."""
    descriptions = [
        "Write a function to calculate factorial",
        "Implement binary search",
        "Create a REST API for user management",
        "Design a distributed cache system",
        "Optimize database query performance",
        "Build a real-time chat application",
        "Implement OAuth2 authentication flow"
    ]

    description = draw(st.sampled_from(descriptions))

    # Add random complexity modifiers
    modifiers = draw(st.lists(
        st.sampled_from([
            "with error handling",
            "with comprehensive tests",
            "optimized for performance",
            "with documentation",
            "using design patterns"
        ]),
        max_size=2
    ))

    if modifiers:
        description = f"{description} {' '.join(modifiers)}"

    return Task(
        description=description,
        complexity=None,  # Will be computed
        constraints=draw(st.lists(st.text(min_size=5, max_size=20), max_size=3)),
        examples=draw(st.lists(st.text(min_size=10, max_size=50), max_size=2)),
        metadata={}
    )


@st.composite
def task_morphism_strategy(draw):
    """Generate random task transformations (morphisms in T)."""
    transformations = [
        lambda t: Task(
            description=f"{t.description} (simplified)",
            complexity=t.complexity,
            constraints=t.constraints[:1] if t.constraints else [],
            examples=t.examples,
            metadata=t.metadata
        ),
        lambda t: Task(
            description=f"{t.description} with additional validation",
            complexity=t.complexity,
            constraints=t.constraints + ["input validation required"],
            examples=t.examples,
            metadata=t.metadata
        ),
        lambda t: Task(
            description=f"Refactor: {t.description}",
            complexity=t.complexity,
            constraints=t.constraints,
            examples=t.examples + ["refactoring example"],
            metadata={**t.metadata, 'refactored': True}
        )
    ]

    return draw(st.sampled_from(transformations))


class TestFunctorLaws:
    """
    Property-based tests for Functor categorical laws.

    Functor F: T → P must satisfy:
    1. F(id_T) = id_P (identity law)
    2. F(g ∘ f) = F(g) ∘ F(f) (composition law)
    """

    @pytest.fixture
    def functor(self) -> Functor:
        """Create functor for testing."""
        return create_task_to_prompt_functor()

    @settings(max_examples=1000)
    @given(task=task_strategy())
    def test_functor_identity_law(self, functor: Functor, task: Task):
        """
        Test Functor Identity Law: F(id) = id

        The functor must preserve the identity morphism.
        Applying the identity function before mapping should equal
        applying the identity after mapping.

        Mathematical Definition:
            F(id_T)(t) = id_P(F(t))

        Args:
            functor: The functor F: T → P
            task: Random task from category T
        """
        # Identity morphism in T
        identity_T = lambda x: x

        # Left side: F(id_T)(task) = F(identity_T(task))
        left_side = functor.map_object(identity_T(task))

        # Right side: id_P(F(task)) = F(task)
        right_side = functor.map_object(task)

        # Verify equality
        assert functor._prompts_equal(left_side, right_side), \
            f"Identity law violated: F(id)({task.description[:50]}) ≠ id(F({task.description[:50]}))"

    @settings(max_examples=1000)
    @given(
        task=task_strategy(),
        f=task_morphism_strategy(),
        g=task_morphism_strategy()
    )
    def test_functor_composition_law(
        self,
        functor: Functor,
        task: Task,
        f: Callable[[Task], Task],
        g: Callable[[Task], Task]
    ):
        """
        Test Functor Composition Law: F(g ∘ f) = F(g) ∘ F(f)

        The functor must preserve composition of morphisms.
        Composing morphisms before mapping should equal
        composing mapped morphisms.

        Mathematical Definition:
            F(g ∘ f)(t) = (F(g) ∘ F(f))(t)

        Args:
            functor: The functor F: T → P
            task: Random task from category T
            f: Random morphism T → T
            g: Random morphism T → T
        """
        # Left side: F(g ∘ f)(task)
        # Compose f and g in category T first
        composed_in_T = lambda t: g(f(t))
        left_side = functor.map_object(composed_in_T(task))

        # Right side: (F(g) ∘ F(f))(task)
        # Map f and g separately, then compose in category P
        f_mapped = functor.map_morphism(f)
        g_mapped = functor.map_morphism(g)

        # Apply f_mapped to F(task), then g_mapped to result
        intermediate = f_mapped(functor.map_object(task))
        right_side = g_mapped(intermediate)

        # Verify equality (structurally, since prompts may differ slightly)
        assert functor._prompts_equal(left_side, right_side), \
            f"Composition law violated: F(g∘f) ≠ F(g)∘F(f) for task: {task.description[:50]}"

    @settings(max_examples=100)
    @given(task=task_strategy())
    def test_functor_preserves_structure(self, functor: Functor, task: Task):
        """
        Test that functor preserves task structure.

        Verifies that:
        - Task description is incorporated into prompt template
        - Task complexity influences prompt strategy
        - Task constraints appear in prompt context

        This is not a categorical law, but validates correct implementation.
        """
        prompt = functor.map_object(task)

        # Verify structure preservation
        assert task.description in prompt.template or task.description in str(prompt.context), \
            "Functor must incorporate task description into prompt"

        assert prompt.context is not None, \
            "Functor must create prompt context"

        assert 'task' in prompt.context, \
            "Functor must include original task in context"

    @settings(max_examples=100)
    @given(task=task_strategy())
    def test_runtime_law_verification(self, functor: Functor, task: Task):
        """
        Test runtime verification methods.

        Verifies that the functor's built-in law verification
        methods correctly validate categorical properties.
        """
        # Verify identity law using runtime method
        assert functor.verify_identity_law(task), \
            f"Runtime identity verification failed for: {task.description[:50]}"

        # Verify composition law using runtime method
        f = lambda t: Task(
            description=f"Enhanced: {t.description}",
            complexity=t.complexity,
            constraints=t.constraints,
            examples=t.examples,
            metadata=t.metadata
        )
        g = lambda t: Task(
            description=f"Optimized: {t.description}",
            complexity=t.complexity,
            constraints=t.constraints,
            examples=t.examples,
            metadata=t.metadata
        )

        assert functor.verify_composition_law(task, f, g), \
            f"Runtime composition verification failed for: {task.description[:50]}"


class TestFunctorImplementation:
    """
    Additional tests for functor implementation details.

    Not categorical laws, but important for correctness.
    """

    @pytest.fixture
    def functor(self) -> Functor:
        """Create functor for testing."""
        return create_task_to_prompt_functor()

    @settings(max_examples=100)
    @given(task=task_strategy())
    def test_complexity_analysis_integration(self, functor: Functor, task: Task):
        """Test that functor correctly integrates complexity analysis."""
        prompt = functor.map_object(task)

        # Verify complexity was analyzed
        assert 'complexity' in prompt.context, \
            "Functor must analyze task complexity"

        complexity = prompt.context['complexity']
        assert isinstance(complexity, ComplexityAnalysis), \
            "Complexity must be ComplexityAnalysis instance"

        assert 0.0 <= complexity.overall <= 1.0, \
            "Complexity must be in range [0.0, 1.0]"

    @settings(max_examples=100)
    @given(task=task_strategy())
    def test_strategy_selection_integration(self, functor: Functor, task: Task):
        """Test that functor correctly selects strategy based on complexity."""
        prompt = functor.map_object(task)

        # Verify strategy was selected
        assert 'strategy' in prompt.context, \
            "Functor must select meta-prompting strategy"

        strategy = prompt.context['strategy']

        # Verify strategy is appropriate for complexity
        complexity = prompt.context['complexity'].overall

        if complexity < 0.3:
            assert strategy.name == 'Direct Execution', \
                "Low complexity should use Direct Execution"
        elif complexity < 0.7:
            assert strategy.name == 'Multi-Approach Synthesis', \
                "Medium complexity should use Multi-Approach Synthesis"
        else:
            assert strategy.name == 'Autonomous Evolution', \
                "High complexity should use Autonomous Evolution"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
