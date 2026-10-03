import asyncio
import tempfile
import unittest
from pathlib import Path

from blackbox.replay import (
    ReplayDivergence,
    ReplayEngine,
    override_output,
    patch_prompt,
    patch_tool_args,
    patch_tool_result,
    swap_model,
)
from blackbox.sdk import Recorder


class DagTools:
    def __init__(self):
        self.calls = {"a": 0, "b": 0, "c": 0, "d": 0}

    def a(self):
        self.calls["a"] += 1
        return 1

    def b(self, a):
        self.calls["b"] += 1
        return a + 1

    def c(self, a):
        self.calls["c"] += 1
        return a + 9

    def d(self, b, c):
        self.calls["d"] += 1
        return b + c


class EchoLLM:
    def __init__(self):
        self.requests = []

    async def chat(self, **request):
        self.requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "content": f"{request['model']}:{request['messages'][0]['content']}"
                    },
                    "finish_reason": "stop",
                }
            ]
        }


def dag_agent(tools: DagTools):
    async def execute(run):
        with run.step("A/tool#1", "tool", "A"):
            run.state["a"] = await run.tool(tools.a)
        with run.step("B/tool#1", "tool", "B"):
            run.state["b"] = await run.tool(tools.b, a=run.state["a"])
        with run.step("C/tool#1", "tool", "C"):
            run.state["c"] = await run.tool(tools.c, a=run.state["a"])
        with run.step("D/tool#1", "tool", "D"):
            total = await run.tool(tools.d, b=run.state["b"], c=run.state["c"])
            run.state["total"] = total
        run.set_outcome(total <= 10, score=float(total), reason=f"total={total}")

    return execute


async def record_base(recorder: Recorder, agent):
    with recorder.run("dag", "dag-task", 100, run_id="base") as run:
        await agent(run)


class ReplayTests(unittest.TestCase):
    def test_no_edit_is_fully_cached_and_preserves_final_state(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                before_calls = dict(tools.calls)
                batch = asyncio.run(ReplayEngine(recorder).replay("base", agent))
                base_last = recorder.database.one(
                    "SELECT state_after FROM steps WHERE run_id = 'base' ORDER BY seq DESC LIMIT 1"
                )

                self.assertEqual(tools.calls, before_calls)
                self.assertEqual(batch.cached_steps, 4)
                self.assertEqual(batch.reexecuted_steps, 0)
                self.assertEqual(set(batch.edited[0].statuses.values()), {"cached"})
                self.assertEqual(batch.edited[0].state_after, base_last["state_after"])
                self.assertEqual(batch.edited[0].outcome, "failed")
            finally:
                recorder.close()

    def test_cone_reexecutes_only_edit_and_true_descendants_with_early_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                events = []
                fixed = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        agent,
                        edits=[override_output("B/tool#1", 0, known_good=True)],
                        event_callback=events.append,
                    )
                )
                statuses = fixed.edited[0].statuses
                self.assertEqual(fixed.invalidated, {"B/tool#1", "D/tool#1"})
                self.assertEqual(statuses["A/tool#1"], "cached")
                self.assertEqual(statuses["B/tool#1"], "edited")
                self.assertEqual(statuses["C/tool#1"], "cached")
                self.assertEqual(statuses["D/tool#1"], "live")
                self.assertEqual(fixed.edited[0].outcome, "passed")
                self.assertTrue(
                    any(
                        event["event"] == "step"
                        and event["data"].get("cache_status") == "invalidated"
                        for event in events
                    )
                )

                same = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base", agent, edits=[override_output("B/tool#1", 2)]
                    )
                )
                self.assertEqual(same.edited[0].statuses["B/tool#1"], "edited")
                self.assertEqual(same.edited[0].statuses["D/tool#1"], "cached")
                self.assertEqual(same.edited[0].outcome, "failed")
            finally:
                recorder.close()

    def test_prefix_reexecutes_every_step_after_the_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                batch = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        agent,
                        edits=[override_output("B/tool#1", 0)],
                        mode="prefix",
                    )
                )
                self.assertEqual(
                    batch.edited[0].statuses,
                    {
                        "A/tool#1": "cached",
                        "B/tool#1": "edited",
                        "C/tool#1": "live",
                        "D/tool#1": "live",
                    },
                )
            finally:
                recorder.close()

    def test_full_mode_and_argument_patch_make_live_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                full = asyncio.run(ReplayEngine(recorder).replay("base", agent, mode="full"))
                self.assertEqual(set(full.edited[0].statuses.values()), {"live"})

                patched = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        agent,
                        edits=[patch_tool_args("B/tool#1", {"a": -1})],
                    )
                )
                self.assertEqual(patched.edited[0].statuses["B/tool#1"], "live")
                self.assertEqual(patched.edited[0].statuses["C/tool#1"], "cached")
                self.assertEqual(patched.edited[0].statuses["D/tool#1"], "live")
                self.assertEqual(patched.edited[0].outcome, "passed")

                result_patch = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base", agent, edits=[patch_tool_result("B/tool#1", 0)]
                    )
                )
                self.assertEqual(result_patch.edited[0].statuses["B/tool#1"], "edited")
                self.assertEqual(result_patch.edited[0].outcome, "passed")
            finally:
                recorder.close()

    def test_prompt_and_model_edits_change_the_exact_request(self):
        with tempfile.TemporaryDirectory() as directory:
            client = EchoLLM()
            recorder = Recorder(directory, mode="live", llm_client=client)

            async def chat_agent(run):
                with run.step("writer/chat#1", "llm", "writer"):
                    response = await run.chat(
                        [{"role": "user", "content": "original"}], model="base-model"
                    )
                    run.state["answer"] = response["choices"][0]["message"]["content"]
                run.set_outcome(True)

            try:
                with recorder.run("chat", "chat-task", 4, run_id="base") as run:
                    asyncio.run(chat_agent(run))
                prompt_batch = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        chat_agent,
                        edits=[patch_prompt("writer/chat#1", "patched instruction")],
                    )
                )
                model_batch = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        chat_agent,
                        edits=[swap_model("writer/chat#1", "new-model")],
                    )
                )
                self.assertEqual(prompt_batch.edited[0].statuses["writer/chat#1"], "live")
                self.assertEqual(
                    client.requests[-2]["messages"][0],
                    {"role": "system", "content": "patched instruction"},
                )
                self.assertEqual(model_batch.edited[0].statuses["writer/chat#1"], "live")
                self.assertEqual(client.requests[-1]["model"], "new-model")
            finally:
                recorder.close()

    def test_k5_fix_beats_paired_control_and_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                batch = asyncio.run(
                    ReplayEngine(recorder).replay(
                        "base",
                        agent,
                        edits=[override_output("B/tool#1", 0, known_good=True)],
                        samples=5,
                        control=True,
                    )
                )
                self.assertEqual(batch.fix_pass_rate, 1.0)
                self.assertEqual(batch.control_pass_rate, 0.0)
                self.assertGreater(batch.fix_interval[0], batch.control_interval[1])
                self.assertEqual(batch.verdict, "VERIFIED")
                fork = recorder.database.one(
                    "SELECT * FROM forks WHERE fork_id = ?", (batch.fork_id,)
                )
                self.assertEqual(fork["verdict"], "VERIFIED")
                self.assertEqual(len(batch.edited), 5)
                self.assertEqual(len(batch.controls), 5)
            finally:
                recorder.close()

    def test_tampered_recorded_output_stops_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = DagTools()
            recorder = Recorder(directory, mode="live")
            try:
                agent = dag_agent(tools)
                asyncio.run(record_base(recorder, agent))
                step = recorder.database.one(
                    "SELECT output_hash FROM steps WHERE run_id = 'base' AND addr = 'A/tool#1'"
                )
                blob = Path(directory) / "content" / "blobs" / f"{step['output_hash']}.json"
                blob.write_text("{}", encoding="utf-8")
                with self.assertRaises(ReplayDivergence):
                    asyncio.run(ReplayEngine(recorder).replay("base", agent))
            finally:
                recorder.close()

    def test_multiple_recorded_values_in_one_step_replay_from_exact_cassette(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory, mode="live")

            async def value_agent(run):
                with run.step("values/state#1", "value", "values"):
                    values = [await run.now(), await run.uuid(), await run.random()]
                    run.state["values"] = values
                run.set_outcome(True)

            try:
                with recorder.run("values", "values-task", 9, run_id="base") as run:
                    asyncio.run(value_agent(run))
                batch = asyncio.run(ReplayEngine(recorder).replay("base", value_agent))
                self.assertEqual(batch.edited[0].statuses, {"values/state#1": "cached"})
                self.assertEqual(batch.reexecuted_steps, 0)
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
