# Contributing to Categorical Meta-Prompting Framework

Thank you for your interest in contributing! This project combines rigorous category theory with practical AI engineering.

## 🎯 Contribution Areas

### 1. Research Contributions (Stream A)
- Analyze academic papers on categorical AI
- Extract categorical structures (functors, monads, comonads)
- Create formal mappings between theory and implementation

### 2. Implementation Contributions (Stream B)
- Test categorical libraries (Effect-TS, DSPy, DisCoPy)
- Create integration examples
- Benchmark on consumer hardware

### 3. Formalization Contributions (Stream C)
- Formal semantics for meta-prompting operations
- Proof sketches for categorical laws
- Type-theoretic foundations

### 4. Code Contributions (Stream D)
- Extract patterns from categorical repositories
- Create reusable abstractions
- Improve test coverage

## 📋 Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/categorical-meta-prompting.git
cd categorical-meta-prompting

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Making Changes

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow coding standards below
3. **Test**: Ensure all tests pass
4. **Commit**: Use conventional commits (see below)
5. **Push**: `git push origin feature/your-feature-name`
6. **Pull Request**: Create PR with detailed description

## 📝 Coding Standards

### Python

- **Style**: Follow PEP 8
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Use Google-style docstrings
- **Tests**: Property-based tests for categorical laws

```python
from typing import Callable, TypeVar

T = TypeVar('T')
P = TypeVar('P')

def functor_map_object(task: T) -> P:
    """Map task to prompt using functor F.

    Args:
        task: Input task of type T

    Returns:
        Prompt of type P

    Raises:
        ValueError: If task is invalid
    """
    ...
```

### TypeScript

- **Style**: Follow Airbnb TypeScript style guide
- **Types**: Use strict mode
- **Composition**: Prefer `pipe` and `Effect.flatMap`
- **Tests**: Use `fast-check` for property-based tests

```typescript
import { Effect, pipe } from 'effect'

const generatePrompt = (task: Task): Effect.Effect<Prompt, Error, AIService> =>
  Effect.gen(function* () {
    const analysis = yield* analyzeTask(task)
    return { template: '...', variables: {} }
  })
```

## ✅ Commit Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add comonad implementation with verified laws
fix: correct functor composition in recursive improvement
docs: update README with integration roadmap
test: add property-based tests for monad laws
refactor: simplify quality monitoring module
```

## 🧪 Testing Requirements

### Categorical Laws

All categorical structures **must** have property-based tests:

```python
import hypothesis.strategies as st
from hypothesis import given

@given(st.builds(Task))
def test_functor_identity_law(task):
    """F(id) = id"""
    functor = create_task_to_prompt_functor(llm)
    assert functor.verify_identity_law(task)

@given(st.builds(Task), st.functions(), st.functions())
def test_functor_composition_law(task, f, g):
    """F(g ∘ f) = F(g) ∘ F(f)"""
    functor = create_task_to_prompt_functor(llm)
    assert functor.verify_composition_law(task, f, g)
```

### Coverage

- **Minimum**: 80% overall
- **Categorical modules**: 95% (critical correctness)
- **Integration tests**: All public APIs

## 📊 Quality Thresholds

| Component | Quality Threshold | Enforcement |
|-----------|------------------|-------------|
| Categorical Laws | 100% pass rate | CI blocking |
| Test Coverage | ≥80% | CI warning |
| Type Safety | 100% (strict mode) | CI blocking |
| Benchmarks | ≥0.90 quality score | Manual review |

## 🔬 Research Contributions

### Adding Papers (Stream A)

1. **Create analysis document**: `stream-a-theory/analysis/author-paper-title.md`
2. **Extract categorical structures**: Identify functors, monads, etc.
3. **Map to implementation**: Connect theory to practice
4. **Update synthesis**: Link in `stream-synthesis/convergence-maps/`

### Implementation Examples (Stream B)

1. **Test on consumer hardware**: Document specs (RAM, GPU)
2. **Benchmark**: Measure latency, throughput, quality
3. **Integration guide**: Step-by-step instructions
4. **Cost analysis**: Estimate $/month for production use

## 🤝 Code Review Process

### For Reviewers

- **Correctness**: Verify categorical laws hold
- **Performance**: Check benchmarks meet thresholds
- **Documentation**: Ensure clear explanations
- **Tests**: Validate property-based coverage

### For Contributors

- **Self-review**: Test locally before pushing
- **Documentation**: Update README if needed
- **Breaking changes**: Discuss in issue first
- **Large PRs**: Break into smaller incremental changes

## 📞 Communication

- **Issues**: Bug reports, feature requests
- **Discussions**: Research questions, design decisions
- **Pull Requests**: Code contributions with detailed context
- **Email**: For private/sensitive matters

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in research papers
- Credited in release notes

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Questions?** Open a [GitHub Discussion](https://github.com/yourusername/categorical-meta-prompting/discussions)

Thank you for helping build rigorous foundations for AI meta-prompting! 🚀
