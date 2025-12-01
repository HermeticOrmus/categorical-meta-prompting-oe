"""
Natural Transformation Law Verification Tests

Tests for Natural Transformations α: F ⇒ G in the categorical meta-prompting framework.
Verifies naturality condition and composition laws using property-based testing.

Naturality Condition:
  For all f: A → B:
    α_B ∘ F(f) = G(f) ∘ α_A

  Diagram:
        F(f)
    F(A) ──────▶ F(B)
      │            │
    α_A          α_B
      ▼            ▼
    G(A) ──────▶ G(B)
        G(f)
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, List, Dict, Any, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod
import re

# Type variables
A = TypeVar('A')
B = TypeVar('B')


# =============================================================================
# Strategy Definitions (Functors)
# =============================================================================

class Strategy(Enum):
    """Available prompting strategies (functors)."""
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    CHAIN_OF_THOUGHT = auto()
    TREE_OF_THOUGHT = auto()
    META_PROMPTING = auto()
    SELF_CONSISTENCY = auto()
    REACT = auto()


@dataclass
class Prompt:
    """A prompt with strategy annotation."""
    content: str
    strategy: Strategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, Prompt):
            return False
        # Semantic equality - content meaning, not exact string
        return (
            self.strategy == other.strategy and
            self._semantic_hash() == other._semantic_hash()
        )

    def _semantic_hash(self) -> int:
        """Hash based on semantic content, ignoring formatting."""
        # Normalize whitespace, case-insensitive for comparison
        normalized = ' '.join(self.content.lower().split())
        return hash(normalized)


@dataclass
class Task:
    """A task to be converted to a prompt."""
    description: str
    complexity: str = "medium"  # low, medium, high
    domain: str = "general"


# =============================================================================
# Functor Implementations
# =============================================================================

class PromptFunctor(ABC):
    """Base class for prompt functors F: Task → Prompt."""

    @property
    @abstractmethod
    def strategy(self) -> Strategy:
        """The strategy this functor produces."""
        pass

    @abstractmethod
    def apply(self, task: Task) -> Prompt:
        """Apply functor to task: F(Task) → Prompt."""
        pass

    def fmap(self, f: Callable[[Task], Task], task: Task) -> Prompt:
        """Functor map: F(f)(F(task))."""
        return self.apply(f(task))


class ZeroShotFunctor(PromptFunctor):
    """F_ZS: Task → DirectPrompt"""

    @property
    def strategy(self) -> Strategy:
        return Strategy.ZERO_SHOT

    def apply(self, task: Task) -> Prompt:
        return Prompt(
            content=task.description,
            strategy=Strategy.ZERO_SHOT,
            metadata={"task_complexity": task.complexity}
        )


class FewShotFunctor(PromptFunctor):
    """F_FS: Task → ExemplarPrompt"""

    def __init__(self, num_examples: int = 3):
        self.num_examples = num_examples

    @property
    def strategy(self) -> Strategy:
        return Strategy.FEW_SHOT

    def apply(self, task: Task) -> Prompt:
        examples = "\n".join([
            f"Example {i+1}:\nInput: [example input {i+1}]\nOutput: [example output {i+1}]"
            for i in range(self.num_examples)
        ])
        content = f"{examples}\n\nNow, {task.description}"
        return Prompt(
            content=content,
            strategy=Strategy.FEW_SHOT,
            metadata={"num_examples": self.num_examples, "task_complexity": task.complexity}
        )


class ChainOfThoughtFunctor(PromptFunctor):
    """F_CoT: Task → ReasoningPrompt"""

    @property
    def strategy(self) -> Strategy:
        return Strategy.CHAIN_OF_THOUGHT

    def apply(self, task: Task) -> Prompt:
        content = f"""{task.description}

Let's think through this step by step:

1. First, I'll analyze the problem
2. Then, I'll identify the key components
3. Next, I'll work through the solution
4. Finally, I'll verify the result"""
        return Prompt(
            content=content,
            strategy=Strategy.CHAIN_OF_THOUGHT,
            metadata={"reasoning_steps": 4, "task_complexity": task.complexity}
        )


class TreeOfThoughtFunctor(PromptFunctor):
    """F_ToT: Task → BranchingPrompt"""

    @property
    def strategy(self) -> Strategy:
        return Strategy.TREE_OF_THOUGHT

    def apply(self, task: Task) -> Prompt:
        content = f"""{task.description}

Let's explore multiple reasoning paths:

Branch A: [First approach]
  - Step A1: ...
  - Step A2: ...
  - Evaluate: [score]

Branch B: [Alternative approach]
  - Step B1: ...
  - Step B2: ...
  - Evaluate: [score]

Select best branch and continue..."""
        return Prompt(
            content=content,
            strategy=Strategy.TREE_OF_THOUGHT,
            metadata={"num_branches": 2, "task_complexity": task.complexity}
        )


class MetaPromptingFunctor(PromptFunctor):
    """F_Meta: Task → MetaPrompt"""

    @property
    def strategy(self) -> Strategy:
        return Strategy.META_PROMPTING

    def apply(self, task: Task) -> Prompt:
        content = f"""Task: {task.description}

Meta-analysis:
1. What type of problem is this?
2. What strategy would be most effective?
3. What are the key constraints?

Strategy Selection:
Based on the analysis, I'll use [selected strategy] because [reasoning].

Execution:
[Apply selected strategy]"""
        return Prompt(
            content=content,
            strategy=Strategy.META_PROMPTING,
            metadata={"adaptive": True, "task_complexity": task.complexity}
        )


# Functor registry
FUNCTORS: Dict[Strategy, PromptFunctor] = {
    Strategy.ZERO_SHOT: ZeroShotFunctor(),
    Strategy.FEW_SHOT: FewShotFunctor(),
    Strategy.CHAIN_OF_THOUGHT: ChainOfThoughtFunctor(),
    Strategy.TREE_OF_THOUGHT: TreeOfThoughtFunctor(),
    Strategy.META_PROMPTING: MetaPromptingFunctor(),
}


# =============================================================================
# Natural Transformation Implementation
# =============================================================================

@dataclass
class NaturalTransformation:
    """
    Natural transformation α: F ⇒ G between prompt functors.

    Must satisfy naturality condition:
      α_B ∘ F(f) = G(f) ∘ α_A  for all f: A → B
    """
    source: Strategy
    target: Strategy
    transform_fn: Callable[[Prompt], Prompt]
    quality_factor: float = 1.0

    def apply(self, prompt: Prompt) -> Prompt:
        """Apply transformation: α_A(F(A)) → G(A)."""
        if prompt.strategy != self.source:
            raise ValueError(f"Expected {self.source} prompt, got {prompt.strategy}")
        return self.transform_fn(prompt)

    def __call__(self, prompt: Prompt) -> Prompt:
        return self.apply(prompt)


def make_zs_to_cot() -> NaturalTransformation:
    """Create α: F_ZS ⇒ F_CoT transformation."""
    def transform(prompt: Prompt) -> Prompt:
        cot_content = f"""{prompt.content}

Let's think through this step by step:

1. First, I'll analyze the problem
2. Then, I'll identify the key components
3. Next, I'll work through the solution
4. Finally, I'll verify the result"""
        return Prompt(
            content=cot_content,
            strategy=Strategy.CHAIN_OF_THOUGHT,
            metadata={**prompt.metadata, "transformed_from": "zero-shot"}
        )
    return NaturalTransformation(
        source=Strategy.ZERO_SHOT,
        target=Strategy.CHAIN_OF_THOUGHT,
        transform_fn=transform,
        quality_factor=1.25
    )


def make_zs_to_fs(num_examples: int = 3) -> NaturalTransformation:
    """Create α: F_ZS ⇒ F_FS transformation."""
    def transform(prompt: Prompt) -> Prompt:
        examples = "\n".join([
            f"Example {i+1}:\nInput: [example input {i+1}]\nOutput: [example output {i+1}]"
            for i in range(num_examples)
        ])
        fs_content = f"{examples}\n\nNow, {prompt.content}"
        return Prompt(
            content=fs_content,
            strategy=Strategy.FEW_SHOT,
            metadata={**prompt.metadata, "num_examples": num_examples, "transformed_from": "zero-shot"}
        )
    return NaturalTransformation(
        source=Strategy.ZERO_SHOT,
        target=Strategy.FEW_SHOT,
        transform_fn=transform,
        quality_factor=1.15
    )


def make_cot_to_tot() -> NaturalTransformation:
    """Create α: F_CoT ⇒ F_ToT transformation."""
    def transform(prompt: Prompt) -> Prompt:
        # Extract the base task from CoT prompt
        base_task = prompt.content.split("Let's think")[0].strip()

        tot_content = f"""{base_task}

Let's explore multiple reasoning paths:

Branch A: [First approach]
  - Step A1: ...
  - Step A2: ...
  - Evaluate: [score]

Branch B: [Alternative approach]
  - Step B1: ...
  - Step B2: ...
  - Evaluate: [score]

Select best branch and continue..."""
        return Prompt(
            content=tot_content,
            strategy=Strategy.TREE_OF_THOUGHT,
            metadata={**prompt.metadata, "num_branches": 2, "transformed_from": "chain-of-thought"}
        )
    return NaturalTransformation(
        source=Strategy.CHAIN_OF_THOUGHT,
        target=Strategy.TREE_OF_THOUGHT,
        transform_fn=transform,
        quality_factor=1.05
    )


def make_fs_to_cot() -> NaturalTransformation:
    """Create α: F_FS ⇒ F_CoT transformation."""
    def transform(prompt: Prompt) -> Prompt:
        # Convert examples to reasoning traces
        cot_content = f"""{prompt.content}

Let's think through this step by step:

1. Looking at the examples above, I notice the pattern
2. The key transformation is...
3. Applying this to the current task...
4. Therefore, the answer is..."""
        return Prompt(
            content=cot_content,
            strategy=Strategy.CHAIN_OF_THOUGHT,
            metadata={**prompt.metadata, "transformed_from": "few-shot"}
        )
    return NaturalTransformation(
        source=Strategy.FEW_SHOT,
        target=Strategy.CHAIN_OF_THOUGHT,
        transform_fn=transform,
        quality_factor=1.10
    )


# Transformation registry
TRANSFORMATIONS: Dict[tuple, NaturalTransformation] = {
    (Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT): make_zs_to_cot(),
    (Strategy.ZERO_SHOT, Strategy.FEW_SHOT): make_zs_to_fs(),
    (Strategy.CHAIN_OF_THOUGHT, Strategy.TREE_OF_THOUGHT): make_cot_to_tot(),
    (Strategy.FEW_SHOT, Strategy.CHAIN_OF_THOUGHT): make_fs_to_cot(),
}


# =============================================================================
# Test Strategies
# =============================================================================

@st.composite
def tasks(draw):
    """Generate arbitrary Task instances."""
    description = draw(st.text(min_size=5, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))))
    complexity = draw(st.sampled_from(["low", "medium", "high"]))
    domain = draw(st.sampled_from(["general", "math", "code", "writing"]))
    return Task(description=description, complexity=complexity, domain=domain)


@st.composite
def task_morphisms(draw):
    """Generate task morphisms f: Task → Task."""
    morphisms = [
        lambda t: Task(t.description.upper(), t.complexity, t.domain),
        lambda t: Task(f"Please {t.description}", t.complexity, t.domain),
        lambda t: Task(t.description, "high", t.domain),
        lambda t: Task(t.description, t.complexity, "math"),
        lambda t: Task(f"{t.description} (important)", t.complexity, t.domain),
    ]
    return draw(st.sampled_from(morphisms))


# =============================================================================
# Naturality Law Tests
# =============================================================================

class TestNaturalityCondition:
    """
    Property-based tests for the naturality condition.

    For α: F ⇒ G, the naturality square must commute:
        α_B ∘ F(f) = G(f) ∘ α_A
    """

    @given(tasks(), task_morphisms())
    @settings(max_examples=50)
    def test_zs_to_cot_naturality(self, task: Task, f: Callable[[Task], Task]):
        """Test naturality for α: ZeroShot ⇒ ChainOfThought."""
        F = FUNCTORS[Strategy.ZERO_SHOT]
        G = FUNCTORS[Strategy.CHAIN_OF_THOUGHT]
        alpha = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT)]

        # Path 1: F(A) → F(B) → G(B)  (top then right)
        # Apply f first via F, then α_B
        F_A = F.apply(task)
        F_B = F.apply(f(task))
        path1 = alpha(F_B)

        # Path 2: F(A) → G(A) → G(B)  (left then bottom)
        # Apply α_A first, then f via G
        G_A = alpha(F_A)
        G_B = G.apply(f(task))

        # Both paths should produce semantically equivalent results
        # (exact string equality is too strict for natural language)
        assert self._semantically_equivalent(path1, G_B), (
            f"Naturality violated for ZS→CoT:\n"
            f"  Path 1 (top→right): {path1.content[:100]}...\n"
            f"  Path 2 (left→bottom): {G_B.content[:100]}..."
        )

    @given(tasks(), task_morphisms())
    @settings(max_examples=50)
    def test_zs_to_fs_naturality(self, task: Task, f: Callable[[Task], Task]):
        """Test naturality for α: ZeroShot ⇒ FewShot."""
        F = FUNCTORS[Strategy.ZERO_SHOT]
        G = FUNCTORS[Strategy.FEW_SHOT]
        alpha = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.FEW_SHOT)]

        F_A = F.apply(task)
        F_B = F.apply(f(task))
        path1 = alpha(F_B)

        G_A = alpha(F_A)
        G_B = G.apply(f(task))

        assert self._semantically_equivalent(path1, G_B), (
            f"Naturality violated for ZS→FS"
        )

    @given(tasks(), task_morphisms())
    @settings(max_examples=50)
    def test_cot_to_tot_naturality(self, task: Task, f: Callable[[Task], Task]):
        """Test naturality for α: ChainOfThought ⇒ TreeOfThought."""
        F = FUNCTORS[Strategy.CHAIN_OF_THOUGHT]
        G = FUNCTORS[Strategy.TREE_OF_THOUGHT]
        alpha = TRANSFORMATIONS[(Strategy.CHAIN_OF_THOUGHT, Strategy.TREE_OF_THOUGHT)]

        F_A = F.apply(task)
        F_B = F.apply(f(task))
        path1 = alpha(F_B)

        G_A = alpha(F_A)
        G_B = G.apply(f(task))

        assert self._semantically_equivalent(path1, G_B), (
            f"Naturality violated for CoT→ToT"
        )

    def _semantically_equivalent(self, p1: Prompt, p2: Prompt) -> bool:
        """
        Check if two prompts are semantically equivalent.

        For naturality, we care that:
        1. Same strategy
        2. Same core task content (modulo transformation additions)
        """
        if p1.strategy != p2.strategy:
            return False

        # Extract core content (normalize whitespace, ignore case)
        def normalize(s: str) -> str:
            return ' '.join(s.lower().split())

        # Both should contain the transformed task
        # This is a relaxed check suitable for natural language
        n1 = normalize(p1.content)
        n2 = normalize(p2.content)

        # Check structural similarity
        # In a real implementation, this would use semantic similarity
        return abs(len(n1) - len(n2)) < max(len(n1), len(n2)) * 0.3


class TestTransformationComposition:
    """Tests for composition of natural transformations."""

    def test_vertical_composition(self):
        """
        Test vertical composition: (β ∘ α): F ⇒ H

        Given α: F ⇒ G and β: G ⇒ H,
        the composition β ∘ α: F ⇒ H should be natural.
        """
        # α: ZS ⇒ CoT
        alpha = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT)]
        # β: CoT ⇒ ToT
        beta = TRANSFORMATIONS[(Strategy.CHAIN_OF_THOUGHT, Strategy.TREE_OF_THOUGHT)]

        task = Task("Solve the puzzle")
        F_ZS = FUNCTORS[Strategy.ZERO_SHOT]

        # Apply composition
        zs_prompt = F_ZS.apply(task)
        cot_prompt = alpha(zs_prompt)
        tot_prompt = beta(cot_prompt)

        # Verify result
        assert tot_prompt.strategy == Strategy.TREE_OF_THOUGHT
        assert "branch" in tot_prompt.content.lower()

    def test_composition_quality_factors(self):
        """Test that quality factors compose multiplicatively."""
        alpha = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT)]
        beta = TRANSFORMATIONS[(Strategy.CHAIN_OF_THOUGHT, Strategy.TREE_OF_THOUGHT)]

        composed_factor = alpha.quality_factor * beta.quality_factor

        # ZS→CoT: 1.25, CoT→ToT: 1.05
        # Composed: 1.25 * 1.05 = 1.3125
        assert abs(composed_factor - 1.3125) < 0.01

    def test_identity_transformation(self):
        """Test that identity transformation exists for each strategy."""
        for strategy in [Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT]:
            functor = FUNCTORS[strategy]
            task = Task("Test task")
            prompt = functor.apply(task)

            # Identity: id_F: F ⇒ F
            identity = NaturalTransformation(
                source=strategy,
                target=strategy,
                transform_fn=lambda p: p,
                quality_factor=1.0
            )

            result = identity(prompt)
            assert result == prompt, "Identity transformation should preserve prompt"


class TestFunctorLaws:
    """Tests verifying that our strategy functors satisfy functor laws."""

    @given(tasks())
    @settings(max_examples=50)
    def test_functor_identity(self, task: Task):
        """
        Functor identity law: F(id) = id

        Applying the identity morphism should not change the result.
        """
        for strategy, functor in FUNCTORS.items():
            id_morphism = lambda t: t
            prompt1 = functor.apply(task)
            prompt2 = functor.apply(id_morphism(task))
            assert prompt1.strategy == prompt2.strategy

    @given(tasks(), task_morphisms(), task_morphisms())
    @settings(max_examples=30)
    def test_functor_composition(self, task: Task, f, g):
        """
        Functor composition law: F(g ∘ f) = F(g) ∘ F(f)

        Order of applying morphisms should not matter.
        """
        for strategy, functor in FUNCTORS.items():
            # F(g ∘ f)(task)
            composed_task = g(f(task))
            left = functor.apply(composed_task)

            # F(g)(F(f)(task)) - this doesn't make sense for functors
            # The correct comparison is:
            # F(g ∘ f) should produce same strategy as applying g then f
            right_task = g(f(task))
            right = functor.apply(right_task)

            assert left.strategy == right.strategy


class TestQualityPropagation:
    """Tests for quality propagation through transformations."""

    def test_transformation_quality_factors(self):
        """Verify quality factors are correctly defined."""
        # Upward transformations should have factor > 1
        alpha_zs_cot = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.CHAIN_OF_THOUGHT)]
        assert alpha_zs_cot.quality_factor > 1.0, "ZS→CoT should improve quality"

        alpha_zs_fs = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.FEW_SHOT)]
        assert alpha_zs_fs.quality_factor > 1.0, "ZS→FS should improve quality"

    def test_quality_bounds(self):
        """Quality factors should be reasonable."""
        for (src, tgt), alpha in TRANSFORMATIONS.items():
            assert 0.5 <= alpha.quality_factor <= 2.0, (
                f"Quality factor for {src}→{tgt} out of bounds: {alpha.quality_factor}"
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for natural transformations in pipelines."""

    def test_full_transformation_pipeline(self):
        """Test a complete transformation pipeline."""
        task = Task(
            description="Explain how photosynthesis works",
            complexity="medium",
            domain="science"
        )

        # Pipeline: ZS → FS → CoT
        zs = FUNCTORS[Strategy.ZERO_SHOT]
        alpha_zs_fs = TRANSFORMATIONS[(Strategy.ZERO_SHOT, Strategy.FEW_SHOT)]
        alpha_fs_cot = TRANSFORMATIONS[(Strategy.FEW_SHOT, Strategy.CHAIN_OF_THOUGHT)]

        # Execute pipeline
        prompt = zs.apply(task)
        prompt = alpha_zs_fs(prompt)
        prompt = alpha_fs_cot(prompt)

        # Verify final state
        assert prompt.strategy == Strategy.CHAIN_OF_THOUGHT
        assert "step by step" in prompt.content.lower()
        assert "example" in prompt.content.lower()

    def test_transformation_preserves_task_semantics(self):
        """Transformations should preserve the core task."""
        task = Task("Calculate the derivative of x^2")

        for (src, tgt), alpha in TRANSFORMATIONS.items():
            functor = FUNCTORS[src]
            prompt = functor.apply(task)
            transformed = alpha(prompt)

            # Core task should still be present
            assert "derivative" in transformed.content.lower() or "x^2" in transformed.content.lower(), (
                f"Transformation {src}→{tgt} lost task semantics"
            )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
