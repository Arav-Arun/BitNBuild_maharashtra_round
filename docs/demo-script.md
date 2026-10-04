# Four-minute Black Box demo

## Before the demo

Start the API and web app with `make dev-api` and `make dev-web` in separate terminals. If the local dataset has not been built, run `make dataset` first. Open <http://localhost:3000/new>.

## Screen-by-screen walkthrough

1. **Create a run (0:00–0:40).** Show the editable trip prompt. Select a built-in example or enter a supported trip request, enable **Use an old exchange rate**, then choose **Run and inspect**. Explain that the offline demo uses deterministic synthetic travel data.
2. **Inspect the failure (0:40–1:35).** Point out the recorded task, the workflow graph and the failed final check. Select the leading candidate, usually `fx_rate`, and trace its recorded output into the budget calculation. Say that the ranker gives evidence-backed candidates and can abstain when it lacks confidence.
3. **Test a correction (1:35–2:45).** Review the suggested correction or choose **Verify candidates**. Verification tests a correction against an unchanged control using paired replay samples. Show which steps were reused and which ran again, then read the pass counts and verdict.
4. **Compare the runs (2:45–3:20).** Open the original and corrected run side by side. Show the changed value, the downstream outcome and the replayed portion of the graph.
5. **Show the evaluation (3:20–4:00).** Open **Results**. On the current frozen evaluation, top-one localization is 71.2% on 111 unseen fault cases versus 58.6% for the best baseline. Replay calls avoided are 55.8%. The natural-failure split has 80 examples but all share the same stale-FX cause, so do not present that score as evidence of broad generalization.

## Short pitch

> Black Box is a flight recorder for AI agents. It links model, tool and state steps into an execution graph, ranks likely failure points with evidence, and lets you test a targeted correction against an unchanged control. On our current synthetic TripCrew evaluation, it localizes the first responsible step in 71.2% of 111 unseen fault cases, compared with a 58.6% baseline.

## Be precise about scope

The prompt runner supports trip requests for the cities shown on the New task page, with dates, 1–6 travellers and an INR budget. The catalog and tools are local deterministic stand-ins; it does not make real bookings. The current evaluation covers one synthetic agent and does not establish cross-agent generalization.
