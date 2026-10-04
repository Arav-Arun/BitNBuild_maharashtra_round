# Five-slide pitch outline

1. **The failure is easy to see; its cause is buried.** Show an agent run with an incorrect budget and many plausible intermediate steps.
2. **A flight recorder built for agents.** Explain exact call records, state snapshots and value-level provenance. State the instrumentation boundary.
3. **Find, change, replay only what depends on it.** Animate the FX quote through its downstream cone. Show cached, invalidated and live states.
4. **A fix needs a control.** Show paired samples, confidence bounds, three verdicts and a generated offline regression check.
5. **Evidence, including limits.** Report dataset splits and intervals. Current ranker on TripCrew: S1 unseen-fault top-1 0.712 (n=111) vs 0.586 for the strongest baseline (paired +0.126, 95% CI [0.063, 0.198]). Say "on one synthetic agent"; do not claim cross-agent generalization, and do not cite S4 (all natural failures share one root).

Add an appendix with the split protocol, both recorder ablations, threat model, and a source ledger for related work before making research comparisons.
