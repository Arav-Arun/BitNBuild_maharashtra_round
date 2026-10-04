# Four-minute Black Box demo

Use `make tripcrew-demo` first, then `make dev-api` and `make dev-web`. Open a failed TripCrew run from the Runs table.

1. **Show the failure (0:00–0:35).** The trip planner exceeded its INR budget. Point out the visible failure in the last step and the ranked suspect list. The failure can surface later than the responsible value.
2. **Follow the value (0:35–1:15).** Select the stale FX quote and show the recorded date, output and downstream provenance edges into the budget and final plan. Mention that diagnosis is a ranked hypothesis and can abstain.
3. **Intervene (1:15–2:15).** Open Fork and fix, replace the quote with a fresh value, and run the paired five-sample replay. The stream marks reused and re-executed steps; unchanged controls use the same seeds.
4. **Read the evidence (2:15–3:00).** Open the fork summary, compare it with the original, and show the earliest divergence, invalidation cone, pass-rate intervals and one of VERIFIED, REFUTED or INCONCLUSIVE.
5. **Make it durable (3:00–3:30).** Export a VERIFIED intervention as an offline pytest artifact and run the generated command.
6. **Close with measured limits (3:30–4:00).** Open Results. State the sample counts and confidence intervals. The current S1 ranker is below the best baseline, so describe the result as a research prototype rather than a win.

## Recorded showcase

When the API is unreachable, the web client falls back to `web/mocks/`. The Results page marks that data as an illustrative fixture. Static fallback supports browsing only; replay requires the local API and a writable dataset.
