# Categorical Meta-Prompting Slash Commands

## Available Commands

| Command | Purpose | Status |
|---------|---------|--------|
| `/meta [task]` | Apply meta-prompting with auto-strategy selection | ✅ Works |
| `/review [file/focus]` | Domain-aware code review | ✅ Works |
| `/debug [error/symptom]` | Systematic debugging protocol | ✅ Works |
| `/rmp [task] [threshold]` | Recursive meta-prompting loop | ✅ Works |
| `/compose [steps...]` | Compose prompt pipeline | ⚠️ Conceptual |
| `/select-prompt [problem]` | Select from registry | ⚠️ Requires setup |
| `/list-prompts [domain]` | List registry contents | ⚠️ Requires setup |

## Honest Assessment

### What Works Now

1. **`/meta`** - Strategy selection (DIRECT/MULTI_APPROACH/AUTONOMOUS_EVOLUTION) based on task complexity. This is pure prompt engineering, no code required.

2. **`/review`** - Domain classification (algorithm, security, API, etc.) with focused review. Works because Claude can read code and apply patterns.

3. **`/debug`** - Systematic REPRODUCE→ISOLATE→HYPOTHESIZE→TEST→FIX protocol. This is a structured prompt that guides Claude's debugging.

4. **`/rmp`** - Recursive improvement loop with quality tracking. Works because it's a prompting pattern, not code execution.

### What Requires Setup

The registry integration commands (`/select-prompt`, `/list-prompts`) require:

1. Python 3.x installed
2. The extensions package importable

**Current limitation:** The `registry_cli.py` has relative imports that require proper package installation. The fallback mode shows static templates.

### What This Demonstrates

1. **Slash commands CAN encode complex meta-prompting strategies** as structured prompts
2. **Domain-aware selection** works through Claude's classification, not code execution
3. **The gap** between static commands and dynamic registry is real - needs proper package setup

### Limitations I'm Being Honest About

1. **Registry requires installation** - The Python bridge doesn't work out of the box due to import structure
2. **Composition is conceptual** - `/compose` describes the idea but doesn't execute real PromptQueue
3. **Quality assessments are self-reported** - No neutral evaluator, so scores should be skeptical
4. **Slash commands are text templates** - They can't do runtime computation

### What Would Make This Actually Work

1. Fix the import structure so `registry_cli.py` works standalone
2. Or: Package the extension properly with `pip install -e .`
3. Or: Rewrite the CLI to not use relative imports

### Novel Value (Honest)

The value is in the **prompting patterns**, not the code:
- Systematic debugging protocol
- Domain-aware review focus
- RMP loop structure with quality tracking
- Strategy selection heuristics

These work because they're encoded in the prompt, not because of dynamic registry lookup.

## Usage

```bash
# Apply meta-prompting
/meta "Implement a rate limiter for an API"

# Domain-aware code review
/review src/auth.py

# Systematic debugging
/debug "TypeError: Cannot read property 'map' of undefined"

# RMP loop with quality threshold
/rmp "Design a caching strategy for user sessions" 8
```
