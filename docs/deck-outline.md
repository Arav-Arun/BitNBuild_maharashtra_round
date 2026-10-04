# Five-slide pitch outline

1. **The failure is easy to see; its cause is buried.** Show an agent run with an incorrect budget and many plausible intermediate steps.
2. **A flight recorder built for agents.** Explain exact call records, state snapshots and value-level provenance. State the instrumentation boundary.
3. **Find, change, replay only what depends on it.** Animate the FX quote through its downstream cone. Show cached, invalidated and live states.
4. **A fix needs a control.** Show paired samples, confidence bounds, three verdicts and a generated offline regression check.
5. **Evidence, including limits.** Report dataset splits and intervals. Current ranker: S1 top-1 0.552 (n=96); strongest baseline 0.625. Natural failures S4 top-1 0.247 (n=85). Avoid a claim that Black Box beats the baseline.

Add an appendix with the split protocol, per-agent HopRAG results, both recorder ablations, threat model, and a source ledger for related work before making research comparisons.
