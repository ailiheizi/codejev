# Core idea: short instructions, fast results, fast decisions

## What the user wants

The big model decides how to do it, where to change, and which fields and variables to use. The
small model receives an explicit instruction and the source text it needs, and quickly assembles
code, an edit, or structured content — it does not chat, it does not write explanations, it does
not plan on its own.

The small model's result goes straight to the big model for judgement. The big model presents only
the result, suggestion or choice the user needs, briefly, so the next step is easy to decide.

**What is saved is repeated thinking, verbose output, and unnecessary handoffs.**

## Three roles

| Role | Only responsible for |
| --- | --- |
| Big model | deciding the approach, sending short instructions, reading results, giving the user a conclusion |
| Small model | writing or editing the result the instruction asks for |
| User | deciding only where a real trade-off is needed |

"Small model does not need to understand" means it carries no business understanding and no
decision responsibility; for syntax and composition it uses what an off-the-shelf code model
already has. There is no need to design a separate thinking process for it.

## An instruction says only what is necessary

Tell the small model the goal, how to change it, and what must be preserved, then give it the
source text it needs. Field names, variable names, interfaces and the approach should come from
the big model wherever possible. Short natural language or plain fields are both fine; no complex
DSL is designed up front.

For example, the big model can say: "修改用户列表函数：只保留 active 为真的项，返回 id 和 name；
保持原顺序，其他不变。" ("Modify the user-list function: keep only rows where active is true,
return id and name; keep the original order, change nothing else.") The small model returns only
the corresponding code or edit.

Short is only acceptable when the information is sufficient; do not make the small model guess
business rules in order to save a few characters. Source text and context that have already been
given and have not changed can be reused.

## What goes back to the user is short too

After reading the artifact, the big model usually gives only the conclusion, the necessary
differences, and a suggestion. Options are given only when there is a real trade-off; several
alternatives or a long report are not the default. Detailed code is expanded on demand.

For example: "Done as requested. Only valid users' id/name are returned; order unchanged. Recommend
accepting." This can only be said after the work is actually finished and checked — it does not
stand in for the real result.

## Current boundary

No error learning, failure-recycling training, reinforcement learning, self-improvement loops, or
complex agent orchestration. When an instruction needs adjusting, the big model simply re-instructs;
that is a change within the current task and does not enter a training loop.

Ordinary SFT stays available as an option: learn only the normal "instruction plus source text →
correct result" and the standard output. First see whether the existing small model can be used
as it is.

The adapter writes or executes the final file or command only after the big model has confirmed it.
The overall value is judged by whether a usable result and a decidable conclusion come faster; no
actual speedup is claimed yet.
