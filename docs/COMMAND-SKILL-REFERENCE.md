# Command & Skill Reference

**Categorical Meta-Prompting Framework**
**Version**: 2.0 | **Date**: 2025-11-30

---

## Table of Contents

1. [Slash Commands](#1-slash-commands)
   - [Meta-Orchestration Layer](#11-meta-orchestration-layer)
   - [Routing Layer](#12-routing-layer)
   - [Object Command Layer](#13-object-command-layer)
2. [Skills](#2-skills)
   - [Core Framework Skills](#21-core-framework-skills)
   - [Implementation Skills](#22-implementation-skills)
   - [Research Skills](#23-research-skills)
3. [Unified Syntax Reference](#3-unified-syntax-reference)
4. [Usage Patterns](#4-usage-patterns)

---

## 1. Slash Commands

### Command Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: META-ORCHESTRATION (Workflow Functors)                    │
│                                                                      │
│  /meta-build    Full feature construction pipeline                  │
│  /meta-fix      Debug → Analyze → Fix → Verify workflow            │
│  /meta-review   Parallel multi-dimensional code review             │
│  /meta-test     Comprehensive testing with coverage                │
│  /meta-refactor Behavior-preserving transformation                 │
│  /meta-deploy   Staged deployment with rollback                    │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: ROUTING & COMPOSITION (Functors + Operators)             │
│                                                                      │
│  /meta          F: Task → Prompt with complexity analysis          │
│  /rmp           M: Prompt →ⁿ Prompt iterative refinement          │
│  /chain         Composition: → || ⊗ >=> operators                  │
│  /route         F: Task → Domain routing                           │
│  /build-prompt  Template × Mode tensor product                     │
│  /compose       Kleisli composition: A → B → C                     │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: OBJECT COMMANDS (Atomic Operations)                       │
│                                                                      │
│  /debug         Systematic debugging protocol                       │
│  /review        Domain-aware code review                           │
│  /template      Step-by-step template construction                 │
│  /select-prompt Prompt selection from registry                     │
│  /list-prompts  Enumerate available prompts                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.1 Meta-Orchestration Layer

These commands orchestrate other commands using the DSL specification.

#### `/meta-build`

**Purpose**: Full orchestrated feature construction

**Categorical Role**: Workflow Functor F_build: Requirement → Implementation

**Syntax**:
```bash
/meta-build "feature description"
```

**Orchestration Pattern**:
```
@orchestration
  @sequential[
    STAGE 1: Codebase Exploration
    STAGE 2: Parallel Design (architecture || data model || interface)
    STAGE 3: Implementation with property testing
    STAGE 4: QA (review || test)
    STAGE 5: Refinement loop until quality >= 0.8
  ]
@end
```

**Example**:
```bash
/meta-build "user authentication with JWT"
```

---

#### `/meta-fix`

**Purpose**: Orchestrated bug resolution workflow

**Categorical Role**: Diagnostic Functor F_fix: Error → Resolution

**Syntax**:
```bash
/meta-fix "error or symptom"
```

**Orchestration Pattern**:
```
@orchestration
  @sequential[
    STAGE 1: Locate & Triage
    STAGE 2: Root Cause Analysis
    STAGE 3: Fix Implementation
    STAGE 4: Verification
    STAGE 5: Documentation
  ]
@end
```

**Example**:
```bash
/meta-fix "TypeError: Cannot read property 'id' of undefined in auth.js:42"
```

---

#### `/meta-review`

**Purpose**: Multi-dimensional parallel code review

**Categorical Role**: Parallel Functor F_review: Code → Reviews × Synthesis

**Syntax**:
```bash
/meta-review "file or changeset"
```

**Orchestration Pattern**:
```
@orchestration
  @parallel[
    → Review: Correctness & Logic
    → Review: Security & Vulnerabilities
    → Review: Performance & Efficiency
    → Review: Maintainability & Style
  ]
  → Synthesize findings
  → Generate report with priorities
@end
```

**Example**:
```bash
/meta-review "src/api/auth.ts"
```

---

#### `/meta-test`

**Purpose**: Comprehensive orchestrated testing

**Categorical Role**: Verification Functor F_test: Code → TestSuite × Results

**Syntax**:
```bash
/meta-test "component or feature"
```

**Orchestration Pattern**:
```
@orchestration
  @sequential[
    STAGE 1: Discover test environment
    STAGE 2: Analyze component
    STAGE 3: @parallel[
      → Generate unit tests
      → Generate integration tests
      → Generate edge case tests
    ]
    STAGE 4: Execute all tests
    STAGE 5: Coverage analysis
  ]
@end
```

**Example**:
```bash
/meta-test "payment processing module"
```

---

#### `/meta-refactor`

**Purpose**: Behavior-preserving code transformation

**Categorical Role**: Isomorphism Functor F_refactor: Code → Code' (structure-preserving)

**Syntax**:
```bash
/meta-refactor "refactoring goal"
```

**Orchestration Pattern**:
```
@orchestration
  @sequential[
    STAGE 1: Baseline (tests, behavior, API surface)
    STAGE 2: Plan refactoring steps
    STAGE 3: @incremental[
      → Apply transformation
      → Run tests
      → Verify behavior preserved
    ]
    STAGE 4: Final verification
  ]
@end
```

**Example**:
```bash
/meta-refactor "extract UserService from monolithic UserController"
```

---

#### `/meta-deploy`

**Purpose**: Staged deployment with rollback support

**Categorical Role**: Production Functor F_deploy: Artifact → Deployment

**Syntax**:
```bash
/meta-deploy "deployment target"
```

**Orchestration Pattern**:
```
@orchestration
  @sequential[
    STAGE 1: Pre-deployment validation
    STAGE 2: Staging deployment
    STAGE 3: Smoke tests
    STAGE 4: Production deployment
    STAGE 5: Health verification
    @fallback: Rollback on failure
  ]
@end
```

**Example**:
```bash
/meta-deploy "v2.1.0 to production"
```

---

### 1.2 Routing Layer

These commands handle task routing, composition, and refinement.

#### `/meta`

**Purpose**: Apply categorical meta-prompting with strategy selection

**Categorical Role**: Functor F: Task → Prompt

**Syntax**:
```bash
/meta @mode:[mode] @tier:[L1-L7] @template:[components] @domain:[domain] "task"
```

**Modifiers**:
| Modifier | Values | Default | Description |
|----------|--------|---------|-------------|
| `@mode:` | active, iterative, dry-run, spec | active | Execution mode |
| `@tier:` | L1-L7 | auto | Complexity tier |
| `@template:` | {context:X}+{mode:Y}+{format:Z} | auto | Template components |
| `@domain:` | ALGORITHM, SECURITY, API, DEBUG, TESTING | auto | Force domain |

**Examples**:
```bash
# Simple - auto-detect everything
/meta "implement rate limiter"

# Iterative with quality target
/meta @mode:iterative @quality:0.85 "optimize algorithm"

# Full specification
/meta @mode:active @tier:L5 @template:{context:expert}+{mode:cot}+{format:code} "design microservices"
```

---

#### `/rmp`

**Purpose**: Recursive Meta-Prompting loop with quality convergence

**Categorical Role**: Monad M: Prompt → Prompt (iterative bind)

**Syntax**:
```bash
/rmp @quality:[0-1] @max_iterations:[n] @mode:[mode] "task"
```

**Modifiers**:
| Modifier | Values | Default | Description |
|----------|--------|---------|-------------|
| `@quality:` | 0.0-1.0 or 0%-100% | 0.8 | Quality threshold |
| `@max_iterations:` | 1-10 | 5 | Maximum iterations |
| `@mode:` | iterative, dry-run, spec | iterative | Execution mode |

**RMP Loop Semantics**:
```
┌─────────────────────────────────────────┐
│ RMP MONADIC LOOP                        │
├─────────────────────────────────────────┤
│ 1. Generate initial prompt (M.unit)     │
│ 2. Execute and evaluate quality         │
│ 3. If quality >= @quality: CONVERGE     │
│ 4. Else: Compute improvement direction  │
│ 5. Apply M.bind(improve) → iterate      │
│ 6. Repeat until converge or max_iter    │
└─────────────────────────────────────────┘
```

**Checkpoint Output**:
```yaml
CHECKPOINT_RMP_[n]:
  iteration: [n]
  quality:
    correctness: 0.85
    clarity: 0.90
    completeness: 0.82
    efficiency: 0.78
    aggregate: 0.84
  quality_delta: +0.05
  status: CONTINUE | CONVERGED | MAX_ITERATIONS
```

**Examples**:
```bash
# Basic quality-gated
/rmp @quality:0.85 "optimize sorting algorithm"

# With iteration limit
/rmp @quality:0.9 @max_iterations:3 "implement secure auth"

# Dry-run to preview
/rmp @mode:dry-run @quality:0.85 "complex task"
```

---

#### `/chain`

**Purpose**: Compose multiple commands using categorical operators

**Categorical Role**: Composition operators: →, ||, ⊗, >=>

**Syntax**:
```bash
/chain @mode:[mode] [composition] "initial input"
```

**Operators**:
| Operator | Unicode | Meaning | Quality Rule |
|----------|---------|---------|--------------|
| `→` | U+2192 | Sequence | min(q₁, q₂) |
| `\|\|` | - | Parallel | mean(q₁, q₂, ...) |
| `⊗` | U+2297 | Tensor | ≤ min(q₁, q₂) |
| `>=>` | - | Kleisli | improves |

**Examples**:
```bash
# Sequential pipeline
/chain [/debug→/fix→/test] "TypeError in auth.py"

# Parallel execution
/chain [/review-security || /review-performance] "audit code"

# Mixed composition
/chain [/analyze→(/design || /prototype)→/implement] "new feature"

# Kleisli with quality gates
/chain [/design>=>implement>=>test] @quality:0.85 "feature"
```

---

#### `/route`

**Purpose**: Route tasks to domain-specific prompts

**Categorical Role**: Functor F_route: Task → Domain → Prompt

**Syntax**:
```bash
/route "task description"
```

**Domain Detection**:
| Pattern | Domain | Route To |
|---------|--------|----------|
| algorithm, complexity, O(n) | ALGORITHM | /meta @domain:ALGORITHM |
| security, auth, injection | SECURITY | /meta @domain:SECURITY |
| API, endpoint, REST | API | /meta @domain:API |
| error, bug, fix | DEBUG | /debug |
| test, coverage, unit | TESTING | /meta-test |

**Example**:
```bash
/route "fix the SQL injection vulnerability in user login"
# → Routes to: /meta @domain:SECURITY "fix SQL injection..."
```

---

#### `/build-prompt`

**Purpose**: Assemble prompts from template components

**Categorical Role**: Tensor product: Context ⊗ Mode ⊗ Format

**Syntax**:
```bash
/build-prompt @template:[components] "task"
```

**Components**:
```
{context:expert}     Expert persona
{context:teacher}    Teaching persona
{context:reviewer}   Reviewer persona
{context:debugger}   Debugger persona

{mode:direct}        Direct answer
{mode:cot}           Chain-of-thought
{mode:multi}         Multi-perspective
{mode:iterative}     Iterative refinement

{format:prose}       Prose output
{format:structured}  Headers/lists
{format:code}        Code with comments
{format:checklist}   Actionable checklist
```

**Example**:
```bash
/build-prompt @template:{context:expert}+{mode:cot}+{format:code} "implement OAuth2"
```

---

#### `/compose`

**Purpose**: Execute a sequence of prompt templates

**Categorical Role**: Kleisli composition: {prompt:A} → {prompt:B} → {prompt:C}

**Syntax**:
```bash
/compose [step1] [step2] [step3] "task"
```

**Available Templates**:
```
{prompt:analyze}   Problem analysis
{prompt:plan}      Implementation planning
{prompt:implement} Code generation
{prompt:test}      Test creation
{prompt:review}    Code review
{prompt:refactor}  Refactoring
```

**Example**:
```bash
/compose analyze plan implement "build caching layer"
```

---

### 1.3 Object Command Layer

Atomic commands that perform specific operations.

#### `/debug`

**Purpose**: Systematic hypothesis-driven debugging

**Categorical Role**: Comonad W.extract: Context → Focus

**Syntax**:
```bash
/debug "error or symptom"
```

**Protocol**:
```
Phase 1: REPRODUCE
  - Establish reliable reproduction
  - Document exact steps
  - Capture error details

Phase 2: ISOLATE
  - Check recent changes
  - Binary search for cause
  - Identify minimal case

Phase 3: HYPOTHESIZE
  - Form 3+ hypotheses
  - Design tests for each
  - Rank by likelihood

Phase 4: FIX
  - Implement targeted fix
  - Verify hypothesis
  - Run regression tests
```

**Example**:
```bash
/debug "Uncaught TypeError: Cannot read property 'map' of undefined"
```

---

#### `/review`

**Purpose**: Domain-aware code review

**Categorical Role**: Functor F_review: Code → Analysis

**Syntax**:
```bash
/review @domain:[domain] "file or focus"
```

**Domain-Specific Focus**:
| Domain | Review Focus |
|--------|--------------|
| ALGORITHM | Complexity, correctness, edge cases |
| API | Error handling, rate limiting, auth |
| SECURITY | OWASP, injection, data exposure |
| DATA | N+1, indexes, consistency |
| UI | Accessibility, performance, state |
| INFRA | Security, idempotency, failures |

**Example**:
```bash
/review @domain:SECURITY "src/auth/login.ts"
```

---

#### `/template`

**Purpose**: Build prompts incrementally with visible states

**Categorical Role**: Exponential object: Context → Prompt

**Syntax**:
```bash
/template "goal"
```

**Process**:
```
Step 1: Base template (empty)
Step 2: Add role/persona
Step 3: Add context
Step 4: Add constraints
Step 5: Add output format
Step 6: Optimize and finalize
```

**Example**:
```bash
/template "create a code review prompt"
```

---

#### `/select-prompt`

**Purpose**: Select best prompt from registry

**Categorical Role**: Coproduct selection: Registry → Prompt

**Syntax**:
```bash
/select-prompt "problem description"
```

**Example**:
```bash
/select-prompt "optimize database queries"
```

---

#### `/list-prompts`

**Purpose**: List available prompts in registry

**Categorical Role**: Hom-set enumeration: Hom(Registry, Prompt)

**Syntax**:
```bash
/list-prompts [domain filter]
```

**Example**:
```bash
/list-prompts ALGORITHM
```

---

## 2. Skills

### Skill Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SKILL HIERARCHY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌──────────────┐                            │
│                         │  meta-self   │ (Identity/Reference)       │
│                         └──────┬───────┘                            │
│                                │                                     │
│            ┌───────────────────┼───────────────────┐                │
│            ▼                   ▼                   ▼                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │     FUNCTOR     │ │     MONAD       │ │   ENRICHMENT    │       │
│  │                 │ │                 │ │                 │       │
│  │ categorical-    │ │ recursive-meta- │ │ quality-        │       │
│  │ meta-prompting  │ │ prompting       │ │ enriched-       │       │
│  │ (F: Task→Prompt)│ │ (M: Prompt→ⁿ)   │ │ prompting ([0,1])│      │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘       │
│           │                   │                   │                 │
│           └───────────────────┼───────────────────┘                │
│                               │                                     │
│                               ▼                                     │
│                    IMPLEMENTATION SKILLS                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.1 Core Framework Skills

#### `meta-self`

**Purpose**: Master reference for unified categorical syntax

**Categorical Role**: Identity morphism / Self-reference protocol

**Use When**:
- Need syntax clarification
- Validating command structure
- Ensuring cross-command consistency

**Key Content**:
- All modifiers (@mode:, @quality:, @tier:, etc.)
- All operators (→, ||, ⊗, >=>)
- Checkpoint format
- Quality assessment structure

**Reference**:
```bash
Skill: "meta-self"
# Provides: Unified syntax specification
```

---

#### `categorical-meta-prompting`

**Purpose**: Full categorical framework (F, M, W, [0,1])

**Categorical Role**: Core categorical structures

**Use When**:
- Implementing functor-based routing
- Building monadic refinement loops
- Extracting context (comonadic)
- Tracking quality degradation

**Key Content**:
- Functor implementation patterns
- Monad laws and verification
- Comonad context extraction
- [0,1]-enriched composition

**Reference**:
```bash
Skill: "categorical-meta-prompting"
# Provides: F, M, W implementations
```

---

#### `recursive-meta-prompting`

**Purpose**: RMP loop implementation

**Categorical Role**: Monad M operations

**Use When**:
- Building iterative refinement
- Quality-gated loops
- Convergence detection

**Key Content**:
- RMP algorithm
- Quality assessment
- Iteration termination
- Plateau detection

**Reference**:
```bash
Skill: "recursive-meta-prompting"
# Provides: M.unit, M.bind, M.join
```

---

#### `quality-enriched-prompting`

**Purpose**: [0,1]-enriched category for quality

**Categorical Role**: Enrichment over quality interval

**Use When**:
- Tracking quality scores
- Implementing quality degradation
- Multi-dimensional assessment

**Key Content**:
- Quality vector structure
- Aggregation formulas
- Threshold semantics
- Tensor product quality

**Reference**:
```bash
Skill: "quality-enriched-prompting"
# Provides: [0,1]-enrichment
```

---

#### `dynamic-prompt-registry`

**Purpose**: Prompt lookup and composition

**Categorical Role**: Reader monad for deferred lookup

**Use When**:
- Dynamic prompt selection
- Template composition
- Registry queries

**Key Content**:
- Registry structure
- Lookup patterns
- @skills: modifier support
- Quality-tracked prompts

**Reference**:
```bash
Skill: "dynamic-prompt-registry"
# Provides: {prompt:X} resolution
```

---

#### `categorical-property-testing`

**Purpose**: Verify categorical laws with property tests

**Categorical Role**: Natural transformation verification

**Use When**:
- Testing functor implementations
- Verifying monad laws
- Checking naturality conditions

**Key Content**:
- fp-ts + fast-check patterns
- Law verification templates
- Property generators
- Counterexample handling

**Reference**:
```bash
Skill: "categorical-property-testing"
# Provides: Law verification utilities
```

---

### 2.2 Implementation Skills

#### `prompt-dsl`

**Purpose**: DSL primitives for prompt composition

**Categorical Role**: Free monad construction

**Use When**:
- Building custom prompt languages
- Implementing typed prompt algebras
- Creating reusable patterns

---

#### `prompt-benchmark`

**Purpose**: Evaluate prompts against benchmarks

**Categorical Role**: Evaluation functor

**Use When**:
- Measuring prompt quality
- Comparing strategies
- Validating improvements

**Supported Benchmarks**:
- MATH
- GSM8K
- Game of 24
- HumanEval
- MMLU

---

#### `effect-ts-ai`

**Purpose**: Effect-TS integration for AI

**Categorical Role**: Effect monad composition

**Use When**:
- Building TypeScript AI pipelines
- Typed error handling
- Provider abstraction

---

#### `dspy-categorical`

**Purpose**: DSPy categorical optimization

**Categorical Role**: Signature functor

**Use When**:
- Declarative LLM programming
- Prompt optimization
- Module composition

---

#### `langgraph-orchestration`

**Purpose**: Multi-agent graph orchestration

**Categorical Role**: State machine functor

**Use When**:
- Building stateful agents
- Cyclic workflows
- Conditional routing

---

#### `lmql-constraints`

**Purpose**: Constrained generation

**Categorical Role**: Constraint category

**Use When**:
- Structured output generation
- Grammar constraints
- Type-safe prompting

---

#### `guidance-grammars`

**Purpose**: Grammar-constrained generation

**Categorical Role**: Grammar category

**Use When**:
- Template-based generation
- Interleaved computation
- Format guarantees

---

#### `mcp-categorical`

**Purpose**: MCP tool composition

**Categorical Role**: Tool category

**Use When**:
- Building MCP servers
- Tool composition patterns
- Context management

---

### 2.3 Research Skills

#### `arxiv-categorical-ai`

**Purpose**: Analyze categorical AI papers

**Use When**:
- Literature review
- Extracting categorical structures
- Building on prior work

**Key Papers**:
- Gavranović et al. (Categorical Deep Learning)
- de Wynter et al. (Meta-Prompting)
- Zhang et al. (Meta Prompting for AI)
- Bradley (Enriched Category Theory of Language)

---

#### `cc2-research-framework`

**Purpose**: CC2.0 seven-function research

**Use When**:
- Systematic research workflows
- Multi-stream coordination
- Categorical research methodology

**Functions**:
- OBSERVE (Comonad extract)
- REASON (Inference functor)
- CREATE (Generative functor)
- ORCHESTRATE (Composition)
- LEARN (Adaptation functor)
- VERIFY (Property testing)
- DEPLOY (Production morphism)

---

#### `discopy-nlp`

**Purpose**: String diagrams for NLP

**Categorical Role**: Monoidal category

**Use When**:
- Compositional semantics
- String diagram computation
- Quantum-inspired NLP

---

#### `polynomial-functors`

**Purpose**: Spivak-Niu polynomial functors

**Use When**:
- Modeling learning systems
- Lens-based composition
- Dynamical systems

---

## 3. Unified Syntax Reference

### 3.1 Command Pattern

```
/<command> @modifier1:value @modifier2:value [composition] "task"
```

### 3.2 All Modifiers

| Modifier | Values | Default | Commands |
|----------|--------|---------|----------|
| `@mode:` | active, iterative, dry-run, spec | active | all |
| `@quality:` | 0.0-1.0 | 0.8 | /rmp, /chain |
| `@tier:` | L1-L7 | auto | /meta, /hekat |
| `@budget:` | integer, [array], auto | auto | /hekat, /task-relay |
| `@variance:` | percentage | 20% | /hekat |
| `@max_iterations:` | integer | 5 | /rmp |
| `@template:` | {context}+{mode}+{format} | auto | /meta, /build-prompt |
| `@domain:` | ALGORITHM, SECURITY, API, DEBUG, TESTING | auto | /meta, /review |
| `@skills:` | discover(), compose(), list | auto | /meta-command |

### 3.3 All Operators

| Operator | Symbol | Meaning | Example |
|----------|--------|---------|---------|
| Sequence | → | A then B | `[A→B→C]` |
| Parallel | \|\| | A and B | `[A\|\|B\|\|C]` |
| Tensor | ⊗ | A with B | `[A⊗B]` |
| Kleisli | >=> | A refine B | `[A>=>B>=>C]` |

---

## 4. Usage Patterns

### 4.1 Simple Task

```bash
/meta "implement rate limiter"
```

### 4.2 Quality-Gated Iteration

```bash
/rmp @quality:0.9 @max_iterations:5 "optimize algorithm"
```

### 4.3 Multi-Stage Pipeline

```bash
/chain [/analyze→/design→/implement→/test] "build feature"
```

### 4.4 Parallel Exploration

```bash
/chain [/approach-a || /approach-b || /approach-c] "evaluate options"
```

### 4.5 Full Orchestration

```bash
/meta-build "user authentication with JWT and refresh tokens"
```

### 4.6 Skill-Driven Command

```bash
/meta-command @skills:discover(domain=API,relevance>0.7) "create endpoint tester"
```

### 4.7 Mixed Composition

```bash
/chain [/analyze→(/design || /prototype)→/implement] @quality:0.85 "new feature"
```

---

*Reference Version: 2.0*
*Generated: 2025-11-30*
*Categorical Foundation: F, M, W, [0,1]-Enriched*
