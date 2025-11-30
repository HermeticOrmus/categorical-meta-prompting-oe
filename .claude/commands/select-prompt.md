---
description: Select the best prompt template from the registry for a given problem
allowed-tools: Bash(python:*), Read
argument-hint: [problem-description]
---

# Dynamic Prompt Selection

## Problem Description
$ARGUMENTS

## Registry Query

Attempting to query the prompt registry for the best match:

!`cd /home/user/categorical-meta-prompting && python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from extensions.dynamic_prompt_registry.registry_cli import cmd_select
    cmd_select('$ARGUMENTS')
except Exception as e:
    print(f'Registry not available: {e}')
    print('Falling back to manual selection.')
" 2>&1 || echo "Python execution failed. Using manual selection."`

## Manual Fallback

If the registry is not available, analyze the problem and select from these templates:

### Available Templates

1. **code_review_algorithm** - For algorithm efficiency review
2. **code_review_security** - For security vulnerability review
3. **debug_systematic** - For systematic debugging
4. **test_generation** - For generating test cases
5. **explain_code** - For code explanation
6. **meta_prompting_direct** - For simple tasks
7. **meta_prompting_multi_approach** - For comparison tasks
8. **meta_prompting_autonomous** - For complex iterative tasks

## Apply Selected Template

Based on the selection (from registry or manual), apply the appropriate template to solve the problem.
