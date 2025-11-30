---
description: List available prompt templates from the registry with quality scores
allowed-tools: Bash(python:*), Read
argument-hint: [domain-filter]
---

# Prompt Registry Listing

## Filter
$ARGUMENTS

## Registry Contents

!`cd /home/user/categorical-meta-prompting && python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from extensions.dynamic_prompt_registry.registry_cli import cmd_list
    cmd_list('$ARGUMENTS' if '$ARGUMENTS' else None)
except Exception as e:
    print('Registry query failed. Showing built-in templates:')
    print()
    print('Available Templates:')
    print('- code_review_algorithm (ALGORITHMS, 0.85)')
    print('- code_review_security (CODE_REVIEW, 0.88)')
    print('- debug_systematic (REASONING, 0.82)')
    print('- test_generation (CODE_GENERATION, 0.84)')
    print('- explain_code (ANALYSIS, 0.80)')
    print('- meta_prompting_direct (GENERAL, 0.75)')
    print('- meta_prompting_multi_approach (REASONING, 0.85)')
    print('- meta_prompting_autonomous (REASONING, 0.90)')
" 2>&1`

## Usage

To use a template, you can:

1. **Direct reference**: `/select-prompt [problem description]`
2. **Compose pipeline**: `/compose analyze plan implement test`
3. **Apply meta-prompting**: `/meta [task description]`
4. **Run RMP loop**: `/rmp [task] [quality-threshold]`
