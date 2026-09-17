# Round 2: differentiable future bond supervision

- Decision: `KEEP`
- Selected arm: `candidate`
- Auxiliary lambda: `0.2303875588749564`
- Added updates per arm: 5000
- Future atom-frame tokens per arm: 142319624

| H | bond relative improvement | systems improved | contact change | amplitude-error change | median motion ratio |
|---|---:|---:|---:|---:|---:|
| H4 | 0.833901 | 8 | 0.390229 | -0.752912 | n/a |
| H8 | 0.838367 | 8 | 0.393200 | -0.722276 | n/a |

free-generation bond error passed the paired threshold without contact, amplitude, or motion-collapse guardrail failure

Remaining risk: one training seed; endpoint-coupled geometry supervision is not a proof of long rollout stability
