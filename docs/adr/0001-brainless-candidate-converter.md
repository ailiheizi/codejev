# ADR 0001: the small model only does brainless candidate selection / code conversion

- Status: Accepted
- Date: 2026-09-20
- Scope: codejev's small-model execution layer

## Context

The goal is not to build another agent that chats, plans and self-repairs. It is to have the big
model break a requirement into clear small tasks, and hand the execution work to a cheap, fast
small model.

Existing measurements show:

- the selection route passes 5/5 on tasks it can express;
- the free-generation route passes 0/5 on comparable real-function tasks;
- the small model has copied a wrong decision because of an example in the prompt;
- when the host guesses the target loop, the sort, or the code structure on the model's behalf,
  the result is a silent error;
- 0.5B is about 2× faster but scores 0/10 on the simplest task, so shrinking the model alone does
  not solve the problem.

The model's responsibility therefore has to be narrowed further.

## Decision

The small model is defined as a **stateless candidate selector / code converter with no business
decisions and no approval authority**:

```text
CodeTask + CandidatePage
→ candidate_id / NO_MATCH
```

Or, once the big model has already fixed the conversion plan:

```text
CodeTask + the necessary source text
→ code body
```

The small model is not responsible for:

- interpreting the user's requirement;
- deciding the operation;
- deciding whether slots such as sort/compare/dedupe are enabled;
- planning the next step;
- diagnosing the cause of a failure;
- choosing file paths;
- setting approval state;
- retrying or training automatically.

## Who owns what

### Big model

The big model turns a natural-language requirement into an explicit `CodeTask`, including:

- the operation;
- the target function / target scope;
- which slots are enabled;
- fields, conditions, sorting and retention constraints;
- whether to split the work into several independent sub-tasks;
- whether `NO_MATCH` is acceptable, or whether to re-instruct / switch to the generation route.

The big model also checks the candidate code or conversion result the small model returns.

### Host

The host owns all determinism and safety boundaries:

- extract candidates;
- assign candidates stable ids;
- validate the candidate page;
- validate the id the small model returns;
- enable or disable slots according to the plan;
- assemble code, produce diffs, run checks;
- bind target path and content hash;
- write to disk only after confirmation.

### Small model

The small model only executes the local task the host and the big model have already fixed. The
smaller the output, the better: for candidate selection, one line with an id; for code conversion,
just the body.

## Why the small model does not judge slots itself

A real regression in the sort slot shows the risk: once the prompt example contained `s=f1,d=desc`,
all 5 instructions that did not ask for sorting got a sort added by the model anyway.

Therefore:

```text
the big model's plan does not enable sort
→ the host does not allow s/d
→ an s/d returned by the small model is rejected too
```

The right to enable an optional slot belongs to the big model's plan and the host, not to the small
model.

### Follow-up: the gate is in place, and the four numbers are measured (2026-09-20)

The sort slot is now implemented in `codejev/decide.py` and is off by default; the right to enable
it stays with the caller, as described above:

```text
Decision(function_id, filter_field, return_fields, sort_field=None, sort_desc=False)
build_decision_prompt(..., sort_enabled=False)   # a protocol bit, not guessed from the instruction
parse_decision(..., sort_enabled=False)          # rejects s/d outright when not enabled
```

The four numbers that item 4 of "Acceptance order" in this document calls for are now in place;
see [measured results, section G](../11-measured-results.md) and
[the slot notes](../12-selection-slots.md): sort tasks 18/18, output tax +15 tokens (+0 for
decisions that do not use the slot), old-task regression 15/15, rejection quality 8/8.

**But this round of measurement exposed a real regression, and it has been fixed**: while adding
the sort slot I also edited the system prompt and deleted the `正确形状示例` ("correct shape
example") line; 1.5B then dropped the `f` key entirely, and the host read "key missing" as "no
filter", so the filter condition was silently deleted (old non-sorting tasks fell from 3/3 to 0/3,
reproducible in the production CLI). The fix was to restore that example line and leave the
disabled-side prohibition in the user message — **the real enforcement is always the host's
`parse_decision`, never the wording of the prompt** — so the safety boundary did not get looser.

This adds a line to the "self-evolution / reuse boundary": **a change such as adding a slot must
come with a paired control group**, because "the model did not pick a key" and "the model
deliberately did not filter" are indistinguishable in the artifact.

## Why the candidate page first, instead of distilling right away

The candidate-page protocol verifies the responsibility boundary without any training:

- the candidates are owned by the host;
- the small model returns only an id;
- `NO_MATCH` goes cleanly back to the big model;
- the candidate code is materialized by the host, byte for byte.

If a real model is unstable on this protocol, the cause of failure can be separated into:

1. the candidate summary / candidate page is unclear;
2. the general Qwen was never trained as a selector;
3. the candidates themselves do not fit the task.

Only once both the protocol and the candidate quality are confirmed correct is it worth training a
smaller scorer. The training target is not to distill a chat model, but to train a dedicated
selection head for `CodeTask + candidate → score/id`.

## The boundary of reuse and "self-evolution"

The allowed evolution is host-level, auditable reuse:

```text
a candidate the big model confirmed as successful
→ added to the candidate library
→ shown/preferred next time
```

Deliberately not done:

- automatic failure-recycling training;
- auto-repair loops;
- automatic entry of unconfirmed online samples into the training set;
- CoT / long thinking-log training;
- complex multi-level agent orchestration.

If we do train later, use only "CodeTask → correct code/candidate" pairs that the big model has
confirmed, for ordinary supervised training or for a dedicated scorer.

## Consequences

### Positive

- the small model can be very small, very fast and very cheap;
- the output format is simple and easy to validate;
- a model error does not by itself gain access to paths, approval, or disk writes;
- `NO_MATCH` is an explicit failure and cannot be disguised as code;
- several independent CodeTasks can be sent in parallel;
- the candidate library can grow gradually, without first building a training platform.

### Cost

- the big model has to break down tasks, make the plan, and check the results;
- candidate extraction and candidate-library quality decide coverage;
- the current Python assembly layer supports Python only;
- when no suitable candidate exists, a generation route or direct big-model handling is still
  required;
- the small model cannot repair an ambiguous task by itself.

## Acceptance order

1. first pass the candidate-page end-to-end acceptance in `docs/10-tomorrow-goal.md`;
2. for the minimal acceptance stage, use `bench/candidate_tasks.json` as the CodeTask input the big
   model has already produced, isolating the remote big-model/API variable;
3. then measure candidate coverage and `NO_MATCH` quality;
4. then add slots such as sort/compare, measuring false triggers for each slot separately;
5. only then decide whether to train a dedicated scorer or build a candidate reuse library.
