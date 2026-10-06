# Real full512 input audit completed after explicit calendar repair

Original full512_v1 audit failed closed when its prerequisite calendar failed.
It is retained unchanged. Sourcea9dc98b adds a separately bound completed-concat
plus repaired-calendar entry,not a relaxation of any data/model qualification.
The new entry test passed1/1,2.28s;12 unrelated cases deselected,not new passes.

Actual full512_v2 completed109.6295s,enter-container/torch-ito,CUDA hidden.
Both seeds:512systems,184320records,8192 exact frozen entries;11 actual scheduled
ClipBatch probes per seed cover largest/first/last/all temporal-H strata.
No crop or missing system,normalization unchanged,no neural calls/updates and
no protected coordinate reads. Data hash bothseeds:
79506f192680908a2058a3237bc8daf4e5191e557ea7a8f0ee671619e466947f.

Schedule71 bd0a2adc2377345162d89d5372dcdb701c773671a46a72c767e112acc01b0bfb
is byte-identical to the unaffected originalfull512 seed71 calendar.
Schedule72 12788f1e72db2dcd66f4b7d9ea70f281b2428154d3b99b9a00d0d73513c0954b
uses the explicitly disclosed one-packing restoration in data-calendar-repair
sourcee86eaf6. See its RESULT_1854 for exact indices,counts and compute confounds.

candidate_input_only_71/72.json are INPUT-ONLY recipes,not training submission.
They intentionally do not contain completed CUDA gate pins. Largest17002-atom
CUDA4/8depth/H4/H8 qualification remains queued; real original16-step control
parity,selected-batch profile and active prefix/cold-resume through this new input
source remain unexecuted. Do not treat CPU completion as expanded-model training.
