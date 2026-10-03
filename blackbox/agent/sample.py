"""Synthetic eight-step fixture, not a trained or model-backed agent."""

from datetime import datetime, timezone
from uuid import uuid4

from blackbox.agent.tasks import TASKS
from blackbox.schema import Labels, Run, Step


def record_sample(store):
    task = TASKS[0]
    run_id = f"sample-{uuid4().hex[:12]}"
    state = {"messages": [], "tool_observations": [], "task_memory": {"task_id": task.task_id}}
    initial = before = store.save_checkpoint(state)
    steps = []
    actions = [
        ("model_call", None, {"question": task.question}, {"action": "retrieve"}),
        (
            "retrieval",
            "retrieve",
            {"task_id": task.task_id},
            {"quantity": task.quantity, "unit_price": task.unit_price},
        ),
        ("state_update", None, {}, {"quantity": task.quantity, "unit_price": task.unit_price}),
        ("model_call", None, {"operation": "multiply"}, {"action": "calculate"}),
        (
            "tool_call",
            "calculate",
            {"left": task.quantity, "right": task.unit_price},
            task.expected,
        ),
        ("state_update", None, {}, task.expected),
        ("model_call", None, {"format": "json"}, task.expected),
        ("tool_call", "write_answer", task.expected, task.expected),
    ]
    for index, (kind, tool, inputs, output) in enumerate(actions):
        started = datetime.now(timezone.utc).isoformat()
        state["messages"].append({"step_index": index, "output": output})
        if tool:
            state["tool_observations"].append({"tool_name": tool, "output": output})
        if kind == "state_update":
            state["task_memory"].update(output)
        if tool == "write_answer":
            state["answer"] = output
        after = store.save_checkpoint(state)
        steps.append(
            Step(
                run_id,
                index,
                index - 1 if index else None,
                kind,
                inputs,
                output,
                tool,
                [str(output)] if kind == "retrieval" else [],
                before,
                after,
                "ok",
                started,
                datetime.now(timezone.utc).isoformat(),
                Labels("success", False),
            )
        )
        before = after
    passed, reason = task.check(state["answer"])
    run = Run(
        run_id,
        task.task_id,
        "success" if passed else "fail",
        reason,
        "synthetic-fixture",
        0,
        initial,
        steps,
    )
    store.save_run(run)
    return run
