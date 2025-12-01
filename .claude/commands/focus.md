---
description: Get immediate focused context (alias for /context @mode:extract @depth:1)
allowed-tools: Read, Grep, Glob
argument-hint: "task"
---

# Focus Command - Immediate Context

**Alias**: This is a convenience alias for `/context @mode:extract @depth:1`

```
W.focus: W(A) → A  (with depth=1)
```

Get the immediate, most relevant context for the current focus.

## Task
$ARGUMENTS

---

## Syntax

```bash
/focus "task"
```

Equivalent to:
```bash
/context @mode:extract @depth:1 "task"
```

## Usage

This is the simplest context extraction - just get what's immediately relevant:

```bash
# What's the current focus?
/focus "what are we working on right now?"

# Get immediate context for a decision
/focus "context for this implementation choice"

# Quick context check
/focus "current state"
```

## When to Use

- **Use `/focus`**: When you need immediate, shallow context
- **Use `/extract`**: When you need configurable depth and focus
- **Use `/context`**: When you need full comonad operations

## Comparison

| Command | Depth | Focus | Operations |
|---------|-------|-------|------------|
| `/focus` | 1 (fixed) | recent (fixed) | extract only |
| `/extract` | configurable | configurable | extract only |
| `/context` | configurable | configurable | extract, duplicate, extend |

---

**See**: `/context` for full comonad operations
**See**: `/extract` for configurable extraction
