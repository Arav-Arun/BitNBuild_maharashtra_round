# Black Box: A Flight Recorder for AI Agents

## Backstory

AI agents solve tasks through long execution graphs involving model calls, tool calls, retrieved context, and changing state. A run may contain many successful steps but fail because of one incorrect intermediate decision. Traditional traces can show what happened, but identifying the responsible step across thousands of executions is difficult.

## Problem Statement

Develop **Black Box**, an AI-powered debugging system that learns from agent execution traces to identify suspicious or failure-causing steps in an agent's execution.

The system should capture observable execution history and train a model to recognize patterns associated with successful and failed runs. When a new execution fails, the system should identify the most likely problematic step and allow that point to be investigated through replay and alternative execution paths.

The system must demonstrate that its learned diagnosis can be evaluated against known failures and that proposed fixes can be tested without unnecessarily repeating unaffected parts of the execution.

## Key Features

- **Execution Data:** Capture relevant information from successful and failed agent runs.
- **Failure Diagnosis:** Train a model to identify anomalous or failure-causing execution steps.
- **Failure Explanation:** Provide evidence from the execution history supporting the diagnosis.
- **Checkpointed Replay:** Allow executions to be investigated from intermediate states.
- **Alternative Execution:** Test changes to a suspected step and observe their effect on the final outcome.
- **Model Evaluation:** Measure the model's ability to localize known failures and generalize to previously unseen failures.
- **Trace Comparison:** Compare executions to understand how changes affected the final outcome.
