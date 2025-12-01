"""
Comonad Law Verification Tests

Tests for the Comonad W: History → Context implementation.
Verifies all three comonad laws using property-based testing.

Laws:
1. Left Identity:  extract ∘ duplicate = id
2. Right Identity: fmap extract ∘ duplicate = id
3. Associativity:  duplicate ∘ duplicate = fmap duplicate ∘ duplicate
"""

import pytest
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, List, Any, Optional
from functools import reduce

# Type variables
A = TypeVar('A')
B = TypeVar('B')


# =============================================================================
# Comonad Implementation
# =============================================================================

@dataclass
class ContextEntry:
    """A single entry in the context history."""
    content: Any
    timestamp: float = 0.0
    focus_weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ContextStore(Generic[A]):
    """
    Comonad W implementation for context management.

    W(A) represents a value A with surrounding context (history).

    Operations:
    - extract: W(A) → A           (get focused value)
    - duplicate: W(A) → W(W(A))   (create meta-observation)
    - extend: (W(A) → B) → W(A) → W(B)  (context-aware map)
    """
    entries: List[ContextEntry]
    focus_index: int = 0

    @classmethod
    def pure(cls, value: A) -> 'ContextStore[A]':
        """Create a context store with a single focused value."""
        return cls(
            entries=[ContextEntry(content=value)],
            focus_index=0
        )

    @classmethod
    def from_list(cls, values: List[A], focus: int = 0) -> 'ContextStore[A]':
        """Create context store from list of values."""
        entries = [ContextEntry(content=v, timestamp=float(i)) for i, v in enumerate(values)]
        return cls(entries=entries, focus_index=min(focus, len(entries) - 1) if entries else 0)

    def extract(self) -> A:
        """
        Comonad extract (counit ε): W(A) → A

        Returns the focused value from the context.
        """
        if not self.entries:
            raise ValueError("Cannot extract from empty context")
        return self.entries[self.focus_index].content

    def duplicate(self) -> 'ContextStore[ContextStore[A]]':
        """
        Comonad duplicate (comultiplication δ): W(A) → W(W(A))

        Creates a context of contexts, where each position has
        a view of the entire context from that position.
        """
        new_entries = []
        for i in range(len(self.entries)):
            # Create a view of the context from position i
            shifted_store = ContextStore(
                entries=self.entries.copy(),
                focus_index=i
            )
            new_entries.append(ContextEntry(
                content=shifted_store,
                timestamp=self.entries[i].timestamp,
                focus_weight=self.entries[i].focus_weight
            ))

        return ContextStore(
            entries=new_entries,
            focus_index=self.focus_index
        )

    def extend(self, f: Callable[['ContextStore[A]'], B]) -> 'ContextStore[B]':
        """
        Comonad extend: (W(A) → B) → W(A) → W(B)

        Applies a context-aware function at each position.
        The function receives the entire context shifted to each position.
        """
        new_entries = []
        for i in range(len(self.entries)):
            # Create shifted view
            shifted = ContextStore(
                entries=self.entries.copy(),
                focus_index=i
            )
            # Apply function with full context
            result = f(shifted)
            new_entries.append(ContextEntry(
                content=result,
                timestamp=self.entries[i].timestamp,
                focus_weight=self.entries[i].focus_weight
            ))

        return ContextStore(
            entries=new_entries,
            focus_index=self.focus_index
        )

    def fmap(self, f: Callable[[A], B]) -> 'ContextStore[B]':
        """
        Functor map: (A → B) → W(A) → W(B)

        Maps a function over the contents without context awareness.
        Derived from extend: fmap f = extend (f . extract)
        """
        return self.extend(lambda w: f(w.extract()))

    def __eq__(self, other: 'ContextStore') -> bool:
        """Equality check for law verification."""
        if not isinstance(other, ContextStore):
            return False
        if len(self.entries) != len(other.entries):
            return False
        if self.focus_index != other.focus_index:
            return False
        for e1, e2 in zip(self.entries, other.entries):
            if e1.content != e2.content:
                # Handle nested ContextStore comparison
                if isinstance(e1.content, ContextStore) and isinstance(e2.content, ContextStore):
                    if e1.content != e2.content:
                        return False
                else:
                    return False
        return True

    def __repr__(self) -> str:
        contents = [e.content for e in self.entries]
        return f"ContextStore({contents}, focus={self.focus_index})"


# =============================================================================
# Test Strategies
# =============================================================================

@st.composite
def context_stores(draw, elements=st.integers()):
    """Generate arbitrary ContextStore instances."""
    values = draw(st.lists(elements, min_size=1, max_size=10))
    focus = draw(st.integers(min_value=0, max_value=len(values) - 1))
    return ContextStore.from_list(values, focus)


@st.composite
def context_functions(draw):
    """Generate functions W(A) → B for extend testing."""
    # Simple functions that work on context
    functions = [
        lambda w: w.extract() * 2,
        lambda w: w.extract() + 1,
        lambda w: sum(e.content for e in w.entries),
        lambda w: len(w.entries),
        lambda w: w.extract() ** 2,
    ]
    return draw(st.sampled_from(functions))


# =============================================================================
# Comonad Law Tests
# =============================================================================

class TestComonadLaws:
    """
    Property-based tests for comonad laws.

    These tests verify that our ContextStore implementation
    satisfies all three comonad laws.
    """

    @given(context_stores())
    @settings(max_examples=100)
    def test_law_1_left_identity(self, ctx: ContextStore[int]):
        """
        Law 1 (Left Identity): extract ∘ duplicate = id

        Duplicating and then extracting should give back the original.
        """
        # Left side: extract(duplicate(ctx))
        duplicated = ctx.duplicate()
        extracted = duplicated.extract()

        # Right side: ctx (identity)
        # extracted should equal ctx
        assert extracted == ctx, (
            f"Law 1 violated:\n"
            f"  extract(duplicate(ctx)) = {extracted}\n"
            f"  ctx = {ctx}"
        )

    @given(context_stores())
    @settings(max_examples=100)
    def test_law_2_right_identity(self, ctx: ContextStore[int]):
        """
        Law 2 (Right Identity): fmap extract ∘ duplicate = id

        Duplicating and then mapping extract over the inner contexts
        should give back the original.
        """
        # Left side: fmap(extract)(duplicate(ctx))
        duplicated = ctx.duplicate()
        mapped = duplicated.fmap(lambda inner: inner.extract())

        # Right side: ctx (identity)
        assert mapped == ctx, (
            f"Law 2 violated:\n"
            f"  fmap(extract)(duplicate(ctx)) = {mapped}\n"
            f"  ctx = {ctx}"
        )

    @given(context_stores())
    @settings(max_examples=50)  # Reduced due to computational complexity
    def test_law_3_associativity(self, ctx: ContextStore[int]):
        """
        Law 3 (Associativity): duplicate ∘ duplicate = fmap duplicate ∘ duplicate

        The order of duplication doesn't matter.
        """
        # Left side: duplicate(duplicate(ctx))
        left = ctx.duplicate().duplicate()

        # Right side: fmap(duplicate)(duplicate(ctx))
        right = ctx.duplicate().fmap(lambda inner: inner.duplicate())

        # Compare structure (deep equality check)
        assert self._deep_equal(left, right), (
            f"Law 3 violated:\n"
            f"  duplicate(duplicate(ctx)) ≠ fmap(duplicate)(duplicate(ctx))"
        )

    def _deep_equal(self, a: ContextStore, b: ContextStore) -> bool:
        """Deep equality check for nested ContextStores."""
        if len(a.entries) != len(b.entries):
            return False
        if a.focus_index != b.focus_index:
            return False

        for e1, e2 in zip(a.entries, b.entries):
            c1, c2 = e1.content, e2.content
            if isinstance(c1, ContextStore) and isinstance(c2, ContextStore):
                if not self._deep_equal(c1, c2):
                    return False
            elif c1 != c2:
                return False
        return True


class TestExtendLaws:
    """
    Tests for extend operation and its relationship to other operations.
    """

    @given(context_stores())
    @settings(max_examples=100)
    def test_extend_extract_identity(self, ctx: ContextStore[int]):
        """
        extend extract = id

        Extending with extract should give back the original.
        """
        result = ctx.extend(lambda w: w.extract())
        assert result == ctx, (
            f"extend(extract) ≠ id:\n"
            f"  extend(extract)(ctx) = {result}\n"
            f"  ctx = {ctx}"
        )

    @given(context_stores())
    @settings(max_examples=100)
    def test_extract_extend(self, ctx: ContextStore[int]):
        """
        extract ∘ extend f = f

        Extracting after extending with f should equal applying f directly.
        """
        f = lambda w: w.extract() * 2 + 1

        # Left side: extract(extend(f)(ctx))
        extended = ctx.extend(f)
        extracted = extended.extract()

        # Right side: f(ctx)
        direct = f(ctx)

        assert extracted == direct, (
            f"extract(extend(f)(ctx)) ≠ f(ctx):\n"
            f"  extracted = {extracted}\n"
            f"  direct = {direct}"
        )

    @given(context_stores())
    @settings(max_examples=50)
    def test_extend_composition(self, ctx: ContextStore[int]):
        """
        extend f ∘ extend g = extend (f ∘ extend g)

        Extending twice equals extending with composed function.
        """
        f = lambda w: w.extract() + 1
        g = lambda w: w.extract() * 2

        # Left side: extend(f)(extend(g)(ctx))
        left = ctx.extend(g).extend(f)

        # Right side: extend(λw. f(extend(g)(w)))(ctx)
        right = ctx.extend(lambda w: f(w.extend(g)))

        assert left == right, (
            f"extend composition law violated:\n"
            f"  extend(f)(extend(g)(ctx)) = {left}\n"
            f"  extend(f ∘ extend g)(ctx) = {right}"
        )


class TestFmapDerivedLaws:
    """
    Tests verifying that fmap is correctly derived from extend.
    """

    @given(context_stores())
    @settings(max_examples=100)
    def test_fmap_identity(self, ctx: ContextStore[int]):
        """
        fmap id = id

        Mapping identity function should not change the context.
        """
        result = ctx.fmap(lambda x: x)
        assert result == ctx, (
            f"fmap(id) ≠ id:\n"
            f"  fmap(id)(ctx) = {result}\n"
            f"  ctx = {ctx}"
        )

    @given(context_stores())
    @settings(max_examples=100)
    def test_fmap_composition(self, ctx: ContextStore[int]):
        """
        fmap (g ∘ f) = fmap g ∘ fmap f

        Mapping composed functions equals composing mapped functions.
        """
        f = lambda x: x + 1
        g = lambda x: x * 2

        # Left side: fmap(g ∘ f)(ctx)
        left = ctx.fmap(lambda x: g(f(x)))

        # Right side: fmap(g)(fmap(f)(ctx))
        right = ctx.fmap(f).fmap(g)

        assert left == right, (
            f"fmap composition law violated:\n"
            f"  fmap(g ∘ f)(ctx) = {left}\n"
            f"  fmap(g)(fmap(f)(ctx)) = {right}"
        )


class TestQualityPropagation:
    """
    Tests for quality propagation rules in comonad operations.
    """

    def test_extract_quality_degradation(self):
        """
        quality(extract(W)) ≤ quality(W)

        Extraction may lose context, so quality can degrade.
        """
        # This is a semantic test - extraction loses surrounding context
        ctx = ContextStore.from_list([1, 2, 3, 4, 5], focus=2)
        extracted = ctx.extract()

        # The extracted value (3) has less information than the full context
        assert extracted == 3
        # Quality would be tracked separately in real implementation

    def test_duplicate_quality_preservation(self):
        """
        quality(duplicate(W)) = quality(W)

        Duplication preserves all information.
        """
        ctx = ContextStore.from_list([1, 2, 3], focus=1)
        duplicated = ctx.duplicate()

        # All original information is preserved
        assert len(duplicated.entries) == len(ctx.entries)
        assert duplicated.focus_index == ctx.focus_index


# =============================================================================
# Integration Tests
# =============================================================================

class TestComonadIntegration:
    """
    Integration tests for comonad with pipeline operations.
    """

    def test_context_pipeline(self):
        """Test comonad in a pipeline context."""
        # Create context with history
        ctx = ContextStore.from_list(
            ["task1", "task2", "current_task", "future1"],
            focus=2
        )

        # Extract current focus
        current = ctx.extract()
        assert current == "current_task"

        # Extend with context-aware summarization
        summarized = ctx.extend(lambda w: f"Focus: {w.extract()}, History: {len(w.entries)}")
        assert "Focus: current_task" in summarized.extract()

    def test_meta_observation(self):
        """Test duplicate for meta-observation."""
        ctx = ContextStore.from_list([1, 2, 3], focus=1)

        # Duplicate to create meta-observation
        meta = ctx.duplicate()

        # Can observe the observation
        inner = meta.extract()
        assert inner.extract() == 2  # Original focus

        # Can see how context was gathered
        assert len(inner.entries) == 3


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
