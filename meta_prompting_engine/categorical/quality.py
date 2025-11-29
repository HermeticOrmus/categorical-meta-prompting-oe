"""
Quality assessment and improvement integration for monad operations.

Provides:
- assess_quality: Output → QualityScore
- extract_improvement: MonadPrompt → Improvement
- integrate_improvement: Prompt × Improvement → EnhancedPrompt
"""

from typing import Dict, Any
from .types import Prompt, QualityScore
import re


def assess_quality(output: str, prompt: Prompt) -> QualityScore:
    """
    Assess quality of LLM output given the prompt.

    Quality Dimensions:
    - Correctness: Does it solve the task?
    - Clarity: Is the solution clear and well-explained?
    - Completeness: Are all aspects addressed?
    - Efficiency: Is the approach optimal?

    Args:
        output: LLM output string
        prompt: Original prompt that generated output

    Returns:
        QualityScore with components breakdown

    Example:
        >>> output = "The maximum is 9."
        >>> quality = assess_quality(output, prompt)
        >>> assert quality.value >= 0.80
    """
    components = {}

    # Correctness (40%): Basic validity checks
    components['correctness'] = _assess_correctness(output, prompt)

    # Clarity (30%): Readability and structure
    components['clarity'] = _assess_clarity(output)

    # Completeness (20%): Addresses all requirements
    components['completeness'] = _assess_completeness(output, prompt)

    # Efficiency (10%): Optimal approach
    components['efficiency'] = _assess_efficiency(output)

    # Weighted average
    weights = {
        'correctness': 0.4,
        'clarity': 0.3,
        'completeness': 0.2,
        'efficiency': 0.1
    }

    overall = sum(components[k] * weights[k] for k in components)

    return QualityScore(
        value=overall,
        components=components
    )


def _assess_correctness(output: str, prompt: Prompt) -> float:
    """
    Assess correctness of output.

    Heuristics:
    - Has a clear answer
    - No contradictions
    - Appropriate length

    Args:
        output: LLM output
        prompt: Original prompt

    Returns:
        Correctness score [0.0, 1.0]
    """
    score = 0.5  # Base score

    # Has content
    if len(output.strip()) > 10:
        score += 0.2

    # Not too short (likely incomplete)
    if len(output) > 50:
        score += 0.1

    # Not too long (likely rambling)
    if len(output) < 2000:
        score += 0.1

    # No error keywords
    error_keywords = ['error', 'cannot', 'unable', 'impossible', 'unclear']
    if not any(kw in output.lower() for kw in error_keywords):
        score += 0.1

    return min(score, 1.0)


def _assess_clarity(output: str) -> float:
    """
    Assess clarity of output.

    Heuristics:
    - Has structure (paragraphs, lists)
    - Uses clear language
    - Proper formatting

    Args:
        output: LLM output

    Returns:
        Clarity score [0.0, 1.0]
    """
    score = 0.5  # Base score

    # Has structure (newlines for paragraphs)
    if output.count('\n\n') >= 1:
        score += 0.15

    # Uses bullet points or numbered lists
    if any(marker in output for marker in ['- ', '* ', '1.', '2.']):
        score += 0.15

    # Reasonable sentence length
    sentences = output.split('.')
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if 10 <= avg_sentence_len <= 30:
        score += 0.1

    # Uses explanatory language
    explanatory_words = ['because', 'therefore', 'thus', 'since', 'so']
    if any(word in output.lower() for word in explanatory_words):
        score += 0.1

    return min(score, 1.0)


def _assess_completeness(output: str, prompt: Prompt) -> float:
    """
    Assess completeness of output.

    Heuristics:
    - Addresses task description
    - Includes examples if requested
    - Covers edge cases

    Args:
        output: LLM output
        prompt: Original prompt

    Returns:
        Completeness score [0.0, 1.0]
    """
    score = 0.6  # Base score

    # Addresses key terms from prompt
    prompt_keywords = extract_keywords(prompt.render())
    output_lower = output.lower()
    keyword_coverage = sum(1 for kw in prompt_keywords if kw in output_lower) / max(len(prompt_keywords), 1)
    score += 0.2 * keyword_coverage

    # Includes examples
    if 'example' in output.lower() or 'e.g.' in output.lower():
        score += 0.1

    # Considers edge cases
    if any(term in output.lower() for term in ['edge case', 'corner case', 'special case']):
        score += 0.1

    return min(score, 1.0)


def _assess_efficiency(output: str) -> float:
    """
    Assess efficiency of approach.

    Heuristics:
    - Mentions complexity
    - Proposes optimal solution
    - Avoids obviously inefficient approaches

    Args:
        output: LLM output

    Returns:
        Efficiency score [0.0, 1.0]
    """
    score = 0.7  # Base score (assume reasonable)

    # Mentions complexity
    if any(term in output.lower() for term in ['o(n)', 'o(log n)', 'complexity', 'efficient']):
        score += 0.15

    # Proposes optimization
    if any(term in output.lower() for term in ['optimize', 'improve', 'faster', 'better']):
        score += 0.1

    # Avoids red flags
    inefficient_keywords = ['nested loop', 'o(n^3)', 'brute force', 'exponential']
    if not any(kw in output.lower() for kw in inefficient_keywords):
        score += 0.05

    return min(score, 1.0)


def extract_keywords(text: str) -> list[str]:
    """
    Extract keywords from text.

    Args:
        text: Input text

    Returns:
        List of significant words (lowercase)
    """
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 3]

    return list(set(keywords))  # Unique keywords


def extract_improvement(monad_prompt) -> Dict[str, Any]:
    """
    Extract improvement suggestions from monad prompt execution.

    This analyzes the output and prompt history to identify
    patterns for improvement.

    Args:
        monad_prompt: MonadPrompt with execution history

    Returns:
        Dictionary with improvement suggestions

    Example:
        >>> improvement = extract_improvement(monad_prompt)
        >>> print(improvement['strategy'])  # 'add_examples'
    """
    improvements = {}

    # Quality-based improvements
    if monad_prompt.quality.value < 0.80:
        improvements['needs_clarity'] = True
        improvements['strategy'] = 'add_structure'

    # Meta-level based improvements
    if monad_prompt.meta_level == 0:
        improvements['add_meta_reflection'] = True
        improvements['strategy'] = 'add_reasoning_steps'

    # History-based improvements
    if len(monad_prompt.history) > 0:
        # Compare to previous iteration
        prev_quality = monad_prompt.history[-1].context.get('quality', 0.5)
        current_quality = monad_prompt.quality.value

        if current_quality <= prev_quality:
            improvements['quality_stagnant'] = True
            improvements['strategy'] = 'try_different_approach'

    return improvements


def integrate_improvement(prompt: Prompt, improvement: Dict[str, Any]) -> Prompt:
    """
    Integrate improvement suggestions into prompt.

    This creates an enhanced prompt based on improvement analysis.

    Args:
        prompt: Original prompt
        improvement: Improvement suggestions from extract_improvement

    Returns:
        Enhanced prompt with improvements integrated

    Example:
        >>> enhanced = integrate_improvement(prompt, {'add_structure': True})
        >>> assert 'Step by step' in enhanced.template
    """
    enhanced_template = prompt.template

    # Apply improvements based on strategy
    strategy = improvement.get('strategy', '')

    if strategy == 'add_structure':
        enhanced_template = f"""Let's approach this systematically:

{enhanced_template}

Provide your solution with clear structure:
1. Analysis
2. Approach
3. Solution
4. Verification"""

    elif strategy == 'add_reasoning_steps':
        enhanced_template = f"""Think step-by-step about this problem:

{enhanced_template}

Show your reasoning process:
- What is the core problem?
- What approach will you use?
- Why is this approach optimal?
- What is the final solution?"""

    elif strategy == 'try_different_approach':
        enhanced_template = f"""Consider alternative approaches to this problem:

{enhanced_template}

Explore multiple solution strategies:
1. Approach A: [describe]
2. Approach B: [describe]
3. Best approach: [select and justify]
4. Final solution: [implement]"""

    # Create enhanced prompt
    return Prompt(
        template=enhanced_template,
        variables=prompt.variables,
        context={
            **prompt.context,
            'improved': True,
            'improvement_strategy': strategy
        },
        meta_level=prompt.meta_level + 1
    )
