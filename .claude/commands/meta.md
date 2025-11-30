---
description: Apply categorical meta-prompting to solve a task with strategy selection based on complexity
allowed-tools: Read, Grep, Glob, Bash(python:*), Edit, Write, TodoWrite
argument-hint: [task-description]
---

# Meta-Prompting Task Solver

You are applying the Categorical Meta-Prompting framework to solve a task.

## Task
$ARGUMENTS

## Strategy Selection

First, analyze the task complexity:

**Complexity Indicators to Check:**
- Simple/direct request → **DIRECT** (complexity < 0.3)
- Needs comparison or multiple options → **MULTI_APPROACH** (0.3-0.7)
- Complex, iterative, needs refinement → **AUTONOMOUS_EVOLUTION** (> 0.7)

Based on your analysis, apply the appropriate strategy:

### If DIRECT (simple tasks):
Just solve it concisely. No overhead needed.

### If MULTI_APPROACH (medium tasks):
1. Generate 2-3 distinct approaches
2. Compare pros/cons of each
3. Synthesize the best solution

### If AUTONOMOUS_EVOLUTION (complex tasks):
1. **Analysis Phase**: What is the core problem? What constraints exist?
2. **Strategy Phase**: What approaches could work? Pick the most promising.
3. **Implementation Phase**: Execute the chosen approach.
4. **Meta-Reflection Phase**: Assess quality (0-10). What's working? What needs improvement?
5. **Synthesis Phase**: Refine based on reflection. Repeat if quality < 8.

## Quality Tracking

After completing the task, provide:
- **Strategy Used**: DIRECT | MULTI_APPROACH | AUTONOMOUS_EVOLUTION
- **Iterations**: How many refinement cycles (if any)
- **Quality Assessment**: Honest 0-10 rating with justification
- **Key Insight**: What made this solution work

## Reference Skills

If you need detailed patterns, invoke:
- `Skill: "recursive-meta-prompting"` for RMP loops
- `Skill: "quality-enriched-prompting"` for quality assessment
- `Skill: "prompt-dsl"` for prompt composition
