# Budget-reduced 500-step delta review request

The independent `REVIEW_FIXED.md` passed commit `07b6ea7` and the three
1,000-step identities. Its own cost finding plus the frozen full evaluation
cost puts that plan beyond the task's total eight-GPU-hour cap. Before observing
any comparison result, the task rule was applied uniformly: every arm is now
500 dynamic updates; cadence remains four, so both static arms have 125 updates.

No code, parent, data, loss, gamma, perturbation, optimizer, seed, or evaluation
coverage changed. Current identities:

- baseline config: `576bde56ebd26650a26fe4804ae9246e86ba94dbeb806dcd300ab43a536343fe`
- internal config: `be6fefec137aca42e65858f286353c164c6ce70a706f8b1e1adffd2350e09f39`
- external config: `4d0815adb6c405108b9e37f62df520f43b6e4f49bb438a85539969740e29d5e7`
- resolved experiment: `d773b5e64467a8ce310aa106b7a6a35911653b046ae14f97b34a254d63ea8188`

The shared target is 125 updates, 739 +/- 3 samples, and 527,677 +/- 3,773
total/supervised atoms. Deterministic plans are internal 737 samples/531,450
atoms and external 742 samples/523,904 atoms.

To keep training plus evaluation within budget, all arms will use the same one
generation seed while retaining all eight validation systems, replicas,
anchors, H/lag buckets, and ordinary/fixed-history views. This reduces sampling
precision and will be reported; it does not asymmetrically change coverage.

Return an explicit delta `PASS` or `FIX_REQUIRED`. Any changed identity requires
another review.
