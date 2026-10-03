"""Fault registry; injectors are implemented with the agent loop in Phase 1."""

from enum import Enum


class FaultType(str, Enum):
    WRONG_TOOL = "wrong_tool"
    CORRUPT_ARGUMENTS = "corrupt_arguments"
    SWAP_RETRIEVAL = "swap_retrieval"
    ALTER_RESULT = "alter_result"
    OVERWRITE_STATE = "overwrite_state"


TRAIN_FAULTS = frozenset(
    {FaultType.WRONG_TOOL, FaultType.CORRUPT_ARGUMENTS, FaultType.SWAP_RETRIEVAL}
)
HELD_OUT_FAULTS = frozenset({FaultType.ALTER_RESULT, FaultType.OVERWRITE_STATE})
