# Black Box demo Q&A

**What does Black Box actually record?** Inputs and outputs for instrumented LLM and tool calls, state snapshots, timing, token counts, errors and producer/consumer edges. It only sees calls made through its SDK.

**Does it inspect the computer or send traces to a hosted service?** No. The default is a local SQLite database and content-addressed blobs. API keys, emails and phone numbers are redacted before storage. An explicit OTLP endpoint can import spans into the same local store.

**Can replay book a real flight or repeat an external side effect?** The included agents use deterministic local tools. For other agents, only instrumented calls can be replayed. Tools with real side effects need an application-provided sandbox or idempotency policy.

**How does it avoid blaming a harmless error?** The ranker uses the full trace and dependency evidence, while deterministic rules provide inspectable findings. The product separates the visible failure from candidate causes and can abstain. A paired unchanged control is part of fix verification.

**What does VERIFIED mean?** The lower 95% Wilson bound for edited samples exceeds the upper bound for paired unchanged controls. It is evidence for the tested task and configuration, not a universal proof.

**Does it beat existing baselines on unseen failures?** On TripCrew, yes: S1 (fault types never seen in training) top-1 is 0.712 (n=111) against 0.586 for the strongest baseline, a paired difference of +0.126 with a 95% interval of [0.063, 0.198]. That is one synthetic agent; cross-agent generalization is not measured. Results displays both with intervals.

**Why trust a model at all?** Diagnosis is a ranked hypothesis that can abstain. Only the paired replay verdict counts as proof, and natural failures (S4) are not yet a meaningful test because every TripCrew natural failure has the same stale-FX root.

**What happens with a new or unsupported trace?** OpenTelemetry traces are imported as read-only. Replay is unavailable because the original application code and deterministic tool adapters are not part of the trace.

**What is required for a public deployment?** The API must receive the recorded SQLite databases, content blobs, and model/evaluation artifacts as immutable build inputs. Public mode stays recorded and has no model-provider secret. Otherwise the web app shows its labelled fixture fallback.
