---
description: Compose a multi-step prompt pipeline from registry templates
allowed-tools: Read, Bash(python:*), Write
argument-hint: [step1] [step2] [step3...]
---

# Prompt Composition

Build a multi-step prompt pipeline by chaining templates.

## Requested Steps
$ARGUMENTS

## Available Composable Units

| Step | Purpose | When to Use |
|------|---------|-------------|
| `analyze` | Understand the problem deeply | Start of complex tasks |
| `plan` | Create structured approach | Before implementation |
| `implement` | Execute the solution | Core work phase |
| `review` | Check for issues | After implementation |
| `test` | Verify correctness | After review |
| `refine` | Iterate based on feedback | If quality < threshold |
| `document` | Explain what was done | End of task |

## Composition Rules

### Sequential (>>)
Steps execute in order, each building on previous output.
```
analyze >> plan >> implement >> test
```

### Parallel (|)
Steps execute independently, results combined.
```
(review_security | review_performance) >> synthesize
```

### Conditional (?)
Branch based on intermediate result.
```
implement >> test >> (pass ? document : refine)
```

## Build the Pipeline

Based on your requested steps, I will:

1. **Parse** the requested composition
2. **Validate** that each step has a known template
3. **Construct** the pipeline in order
4. **Execute** each step, passing context forward
5. **Track** quality at each stage

## Execution

For each step in the pipeline:

```
Step N: [step_name]
─────────────────
Input: [from previous step]
Template: [selected template]
Output: [result]
Quality: [0-10]
```

## Final Output

After all steps complete:

```
Pipeline: step1 >> step2 >> step3
Total Steps: N
Min Quality: X
Final Result: [synthesized output]
```

---

**Note**: This is a conceptual demonstration of prompt composition.
For production use, see the `extensions/dynamic-prompt-registry/queue.py`
implementation of PromptQueue with proper categorical semantics.
