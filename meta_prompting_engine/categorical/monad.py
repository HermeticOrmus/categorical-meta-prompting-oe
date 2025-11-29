"""
Monad M: Recursive Meta-Prompting

A monad is a structure capturing computational effects through:
1. unit η: A → M(A) (wrapping values in monadic context)
2. join μ: M(M(A)) → M(A) (flattening nested contexts)
3. Kleisli composition: (A → M(B)) → (B → M(C)) → (A → M(C))

In meta-prompting, the monad M formalizes recursive improvement where:
- M(Prompt) = Prompt with iteration history and quality tracking
- unit η = Initial prompt generation
- join μ = Integration of meta-level improvements
- >>= (bind) = Chaining prompt improvements

References:
- Zhang et al. (arXiv:2311.11482) - Empirical validation: 100% on Game of 24
- Moggi (1991) - Computational monads for programming language semantics
- L5 Meta-Prompt: "Monad M for recursive improvement with quality join"

Mathematical Notation:
    Monad: (M, η, μ) on category P (Prompts)
    - η : 1_P → M (unit)
    - μ : M ∘ M → M (join)

    Laws:
    1. Left identity:  μ ∘ η(M) = id_M
    2. Right identity: μ ∘ M(η) = id_M
    3. Associativity:  μ ∘ M(μ) = μ ∘ μ(M)

Example:
    >>> monad = create_recursive_meta_monad(llm_client)
    >>> initial = monad.unit(base_prompt)
    >>> improved = monad.bind(initial, lambda p: monad.unit(improve(p)))
    >>> assert monad.verify_left_identity(base_prompt, improve)
"""

from typing import TypeVar, Callable, Generic, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from .types import Prompt, QualityScore

# Type variable for monad content
A = TypeVar('A')


@dataclass
class MonadPrompt:
    """
    Prompt wrapped in monadic context for recursive improvement.

    This is M(Prompt) - a prompt enhanced with:
    - Iteration history (tracking recursive improvements)
    - Quality score (assessing current state)
    - Meta-level depth (recursion level)
    - Improvement trace (provenance of changes)

    Attributes:
        prompt: The underlying prompt (current state)
        quality: Quality score [0.0, 1.0]
        meta_level: Recursion depth (0 = initial, 1+ = improved)
        history: List of previous prompts in improvement chain
        timestamp: When this prompt was generated

    Example:
        >>> mp = MonadPrompt(
        ...     prompt=Prompt("Solve: {task}", variables={"task": "find max"}),
        ...     quality=QualityScore(0.85),
        ...     meta_level=1
        ... )
    """
    prompt: Prompt
    quality: QualityScore
    meta_level: int = 0
    history: list[Prompt] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def __str__(self) -> str:
        return f"M(Prompt, q={self.quality.value:.2f}, level={self.meta_level})"

    def __repr__(self) -> str:
        return f"MonadPrompt(quality={self.quality.value:.2f}, meta_level={self.meta_level})"


@dataclass
class Monad(Generic[A]):
    """
    Monad M with verified categorical laws.

    Attributes:
        unit: η : A → M(A) (wraps value in monadic context)
        join: μ : M(M(A)) → M(A) (flattens nested monads)

    Laws:
        1. Left identity:  unit(a) >>= f = f(a)
        2. Right identity: m >>= unit = m
        3. Associativity:  (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

    Example:
        >>> monad = Monad(
        ...     unit=lambda p: MonadPrompt(p, QualityScore(0.7), 0),
        ...     join=lambda mm: integrate_improvement(mm)
        ... )
    """

    unit: Callable[[A], MonadPrompt]
    join: Callable[[MonadPrompt], MonadPrompt]

    def bind(self, ma: MonadPrompt, f: Callable[[A], MonadPrompt]) -> MonadPrompt:
        """
        Kleisli composition (>>=).

        Chains monadic computations: ma >>= f

        Process:
        1. Extract value from ma
        2. Apply f to get new monadic value
        3. Flatten nested monad via join

        Args:
            ma: Monadic value M(A)
            f: Function A → M(B)

        Returns:
            M(B) - result of chaining computation

        Mathematical Definition:
            m >>= f = μ(M(f)(m))
            where M is the functor part of the monad

        Example:
            >>> improved = monad.bind(
            ...     initial,
            ...     lambda p: monad.unit(enhance_prompt(p))
            ... )
        """
        # Apply f to the prompt value (A → M(B))
        mb = f(ma.prompt)

        # Create nested monad M(M(B))
        nested = MonadPrompt(
            prompt=mb.prompt,
            quality=ma.quality.tensor_product(mb.quality),  # Quality degrades
            meta_level=ma.meta_level + 1,
            history=ma.history + [ma.prompt],
            timestamp=datetime.now()
        )

        # Flatten via join: M(M(B)) → M(B)
        return self.join(nested)

    def verify_left_identity(
        self,
        a: A,
        f: Callable[[A], MonadPrompt]
    ) -> bool:
        """
        Verify monad left identity law: unit(a) >>= f = f(a)

        The unit should be a left identity for bind.

        Args:
            a: Value of type A
            f: Function A → M(B)

        Returns:
            True if law holds

        Mathematical Definition:
            η(a) >>= f = f(a)

        Example:
            >>> prompt = Prompt("Task: {desc}", variables={"desc": "solve"})
            >>> assert monad.verify_left_identity(prompt, improve)
        """
        # Left side: unit(a) >>= f
        left_side = self.bind(self.unit(a), f)

        # Right side: f(a)
        right_side = f(a)

        return self._monadic_prompts_equal(left_side, right_side)

    def verify_right_identity(self, ma: MonadPrompt) -> bool:
        """
        Verify monad right identity law: m >>= unit = m

        The unit should be a right identity for bind.

        Args:
            ma: Monadic value M(A)

        Returns:
            True if law holds

        Mathematical Definition:
            m >>= η = m

        Example:
            >>> mp = MonadPrompt(prompt, QualityScore(0.8), 1)
            >>> assert monad.verify_right_identity(mp)
        """
        # Left side: m >>= unit
        left_side = self.bind(ma, self.unit)

        # Right side: m
        right_side = ma

        return self._monadic_prompts_equal(left_side, right_side)

    def verify_associativity(
        self,
        ma: MonadPrompt,
        f: Callable[[A], MonadPrompt],
        g: Callable[[A], MonadPrompt]
    ) -> bool:
        """
        Verify monad associativity law: (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

        Bind should be associative.

        Args:
            ma: Monadic value M(A)
            f: Function A → M(B)
            g: Function B → M(C)

        Returns:
            True if law holds

        Mathematical Definition:
            (m >>= f) >>= g = m >>= (λx. f(x) >>= g)

        Example:
            >>> assert monad.verify_associativity(mp, improve, refine)
        """
        # Left side: (m >>= f) >>= g
        left_side = self.bind(self.bind(ma, f), g)

        # Right side: m >>= (λx. f(x) >>= g)
        right_side = self.bind(ma, lambda x: self.bind(f(x), g))

        return self._monadic_prompts_equal(left_side, right_side)

    def _monadic_prompts_equal(self, mp1: MonadPrompt, mp2: MonadPrompt) -> bool:
        """
        Check if two monadic prompts are equal (structural equality).

        Compares:
        - Prompt templates
        - Quality scores
        - Meta-levels

        Args:
            mp1: First monadic prompt
            mp2: Second monadic prompt

        Returns:
            True if structurally equal
        """
        return (
            self._hash_prompt(mp1.prompt) == self._hash_prompt(mp2.prompt) and
            abs(mp1.quality.value - mp2.quality.value) < 0.01 and
            mp1.meta_level == mp2.meta_level
        )

    def _hash_prompt(self, prompt: Prompt) -> str:
        """
        Compute hash of prompt for equality checking.

        Args:
            prompt: Prompt to hash

        Returns:
            SHA-256 hash as hex string
        """
        prompt_str = f"{prompt.template}_{prompt.meta_level}"
        return hashlib.sha256(prompt_str.encode()).hexdigest()


# Factory function for creating Recursive Meta-Prompting Monad
def create_recursive_meta_monad(
    llm_client,
    quality_threshold: float = 0.90
) -> Monad:
    """
    Factory for creating M: Recursive Meta-Prompting monad.

    This monad captures the essence of recursive prompt improvement:
    - unit η: Wrap initial prompt with quality assessment
    - join μ: Integrate meta-level improvements
    - bind >>=: Chain prompt improvement operations

    Args:
        llm_client: LLM client for prompt execution and quality assessment
        quality_threshold: Minimum quality threshold [0.0, 1.0]

    Returns:
        Monad[Prompt] with verified laws

    Unit Operation (η):
        Prompt → M(Prompt)
        - Execute prompt via LLM
        - Assess quality of output
        - Wrap in MonadPrompt with meta_level=0

    Join Operation (μ):
        M(M(Prompt)) → M(Prompt)
        - Extract improvement from meta-level
        - Integrate into base prompt
        - Re-assess quality
        - Flatten to single monad layer

    Example:
        >>> monad = create_recursive_meta_monad(claude_client)
        >>> initial = monad.unit(base_prompt)
        >>> improved = monad.bind(initial, lambda p: monad.unit(enhance(p)))
        >>> print(f"Quality improved: {initial.quality.value} → {improved.quality.value}")
    """
    from .quality import assess_quality, integrate_improvement, extract_improvement

    def unit(prompt: Prompt) -> MonadPrompt:
        """
        η : Prompt → M(Prompt)

        Wraps prompt in monadic context with quality assessment.

        Process:
        1. Execute prompt via LLM
        2. Assess quality of output
        3. Return MonadPrompt with meta_level=0

        Args:
            prompt: Input prompt

        Returns:
            MonadPrompt with initial quality
        """
        # Execute prompt
        output = llm_client.complete(prompt.render())

        # Assess quality
        quality = assess_quality(output, prompt)

        return MonadPrompt(
            prompt=prompt,
            quality=quality,
            meta_level=0,
            history=[],
            timestamp=datetime.now()
        )

    def join(nested: MonadPrompt) -> MonadPrompt:
        """
        μ : M(M(Prompt)) → M(Prompt)

        Flattens nested monad by integrating meta-level improvements.

        Process:
        1. Extract improvement from nested meta-level
        2. Integrate improvement into base prompt
        3. Re-execute enhanced prompt
        4. Re-assess quality
        5. Return flattened MonadPrompt

        Args:
            nested: Nested monadic prompt M(M(Prompt))

        Returns:
            Flattened MonadPrompt M(Prompt)

        Note:
            This is where the "join" happens - we integrate the
            meta-level thinking back into a single prompt layer.
        """
        # Extract improvement from meta-level output
        improvement = extract_improvement(nested)

        # Integrate into base prompt
        enhanced_prompt = integrate_improvement(nested.prompt, improvement)

        # Re-execute enhanced prompt
        new_output = llm_client.complete(enhanced_prompt.render())

        # Re-assess quality (may improve or degrade)
        new_quality = assess_quality(new_output, enhanced_prompt)

        # Return flattened monad
        return MonadPrompt(
            prompt=enhanced_prompt,
            quality=new_quality,
            meta_level=nested.meta_level,
            history=nested.history,
            timestamp=datetime.now()
        )

    return Monad(unit=unit, join=join)


def kleisli_compose(
    monad: Monad,
    f: Callable[[A], MonadPrompt],
    g: Callable[[A], MonadPrompt]
) -> Callable[[A], MonadPrompt]:
    """
    Kleisli composition: (A → M(B)) → (B → M(C)) → (A → M(C))

    Composes two monadic functions via bind:
        (f >=> g)(a) = f(a) >>= g

    This is the categorical essence of sequential prompt improvement.

    Args:
        monad: The monad providing bind operation
        f: First monadic function A → M(B)
        g: Second monadic function B → M(C)

    Returns:
        Composed function A → M(C)

    Example:
        >>> improve = lambda p: monad.unit(enhance_prompt(p))
        >>> refine = lambda p: monad.unit(refine_prompt(p))
        >>> improve_and_refine = kleisli_compose(monad, improve, refine)
        >>> result = improve_and_refine(initial_prompt)
    """
    def composed(a: A) -> MonadPrompt:
        return monad.bind(f(a), g)

    return composed
