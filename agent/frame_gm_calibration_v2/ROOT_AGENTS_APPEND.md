# 由主实现session择要合并到根AGENTS（不要全文替换）

## Task authority and review handoff

- Follow the current task's explicit authorization. An older task's pause after tiny/pilot is not a permanent repository requirement when the user has authorized the new task through training and evaluation.
- For tasks requesting independent review, use a fresh session without the implementation conversation, review a fixed code commit and factual evidence, and preserve its written findings. One session owns code fixes; the reviewer does not edit model code.
- Keep review findings, fix responses, code/config identities and training artifacts traceable. Do not treat a review of different source as approval for a new run.
- Repair first, then run checks targeted at the changed numerical or scientific risks. Whole-suite regression is not an automatic prerequisite for CUDA smoke or every experiment.
- Source-sampled training must expose autograd, target boundaries, physical-time units and RNG streams explicitly. Do not call an inference-only no-grad sampler and label the result end-to-end training.
