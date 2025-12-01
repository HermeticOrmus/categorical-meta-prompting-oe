---
description: Extract focused context from history (alias for /context @mode:extract)
allowed-tools: Read, Grep, Glob, Bash(git log:*), Bash(git diff:*)
argument-hint: @focus:[target] @depth:[n] "task"
---

# Extract Command - Comonad Extract Operation

**Alias**: This is a convenience alias for `/context @mode:extract`

```
W.extract: W(A) → A
```

Extract the focused/relevant context from execution history.

## Task
$ARGUMENTS

---

## Syntax

```bash
/extract @focus:[target] @depth:[n] "task"
```

Equivalent to:
```bash
/context @mode:extract @focus:[target] @depth:[n] "task"
```

## Modifiers

| Modifier | Values | Default | Description |
|----------|--------|---------|-------------|
| `@focus:` | recent, all, file, conversation | recent | Context target |
| `@depth:` | 1-10 | 3 | History depth |

## Examples

```bash
# Extract recent context
/extract @focus:recent "what have we been working on?"

# Extract file-specific context
/extract @focus:file "context for current implementation"

# Extract conversation flow
/extract @focus:conversation @depth:5 "how did we decide this?"
```

## Categorical Law

As comonad counit (ε):
```
extract ∘ duplicate = id
```

Extracting after duplicating returns the original context.

---

**See**: `/context` for full comonad operations (extract, duplicate, extend)
