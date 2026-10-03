"""Checkpoint loading; suffix execution and stitching are Phase 2 work."""
from blackbox.recorder import Store


def load_before(store: Store, run_id: str, step_index: int):
    run = store.load_run(run_id)
    if not 0 <= step_index < len(run.steps):
        raise ValueError("step index outside trace")
    step = run.steps[step_index]
    return store.load_checkpoint(step.state_before_ref), step
