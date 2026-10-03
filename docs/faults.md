# Fault injection contract

Each injected fault changes exactly one step and records its index as ground truth.
The registry is defined in `blackbox/agent/faults.py`; injection is not implemented yet.

| Fault | Changed field | Split family |
| --- | --- | --- |
| wrong_tool | tool_name | train / validation / Test A |
| corrupt_arguments | input | train / validation / Test A |
| swap_retrieval | retrieved_context | train / validation / Test A |
| alter_result | output | Test B only |
| overwrite_state | referenced state snapshot | Test B only |

Keep all steps of a run in the same split. Fit successful-run references on train
only. An injected fault is not guaranteed to cause failure: always use the checker
to determine outcome. Never expose injection metadata to feature extraction.
