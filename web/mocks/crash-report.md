# Mumbai → Singapore under ₹1,53,516 failed on a stale exchange rate

Run `tc-0001`. Illustrative fixture.

## Synopsis

tc-0001 failed the checker (Incorrect INR total: got 130949.18, expected 134999.04). The responsible step is fx/tool#1, confirmed by replay: 5/5 edited runs passed against 0/5 unchanged controls.

## Sequence of events

0. `planner/chat#1`: The planner parsed the request into constraints.
5. `fx/tool#1`: The FX tool returned a rate dated 2026-03-03.
11. `budget/tool#1`: The budget step multiplied by that rate.
12. `writer/chat#1`: The writer put the budget total into the plan.
13. `verifier/chat#1`: The verifier passed the plan through unchanged.
15. `checker/state#1`: The checker rejected the plan: Incorrect INR total: got 130949.18, expected 134999.04

## Probable cause

- The SGD to INR rate came from a snapshot months older than the rest of the run's data. (`fx/tool#1/output/as_of`)

## Contributing factors

- Nothing downstream checks the age of the rate before using it. (`budget/tool#1/input/args/fx/as_of`)

## Findings

- The exchange rate is 214 days older than the freshest data in this run. (`fx/tool#1/output/as_of`)
- The rate (54.4) is 15% below the rate used by the nearest passing run. (`fx/tool#1/output/rate`)
- 5 later steps consume this output, so a wrong value reaches the final plan. (`fx/tool#1/output`)

## Recommended fix

- Re-fetch the rate with fresh=true. (`fx/tool#1/input/args`)

## Verification

- VERIFIED at K=5: fix 5/5 (95% interval 0.566 to 1.000) against control 0/5 (0.000 to 0.434). (`fx/tool#1/output`)
