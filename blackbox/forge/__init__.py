"""Fault Forge: synthetic failure injection and labeling for training data."""

from blackbox.forge.inject import FaultInjector
from blackbox.forge.label import ForkLabel, ForkResult, Labeler
from blackbox.forge.natural_label import NaturalLabel, NaturalLabeler
from blackbox.forge.operators import (
    C1InstructionMisread,
    C2ConstraintDropped,
    C3StateCorruption,
    C4RepeatedLoop,
    D1WrongArguments,
    D2WrongTool,
    D3HallucinatedValue,
    D4StopsTooEarly,
    FaultOperator,
    R1IrrelevantDocuments,
    R2PoisonedFact,
    T1WrongValue,
    T2StaleData,
    T3Empty404,
    T4Timeout500,
    T5SchemaDrift,
    all_operators,
    held_out_operators,
    seen_operators,
)
from blackbox.forge.runner import ForgeRunner

__all__ = [
    "C1InstructionMisread",
    "C2ConstraintDropped",
    "C3StateCorruption",
    "C4RepeatedLoop",
    "D1WrongArguments",
    "D2WrongTool",
    "D3HallucinatedValue",
    "D4StopsTooEarly",
    "FaultInjector",
    "FaultOperator",
    "ForkLabel",
    "ForkResult",
    "ForgeRunner",
    "Labeler",
    "NaturalLabel",
    "NaturalLabeler",
    "R1IrrelevantDocuments",
    "R2PoisonedFact",
    "T1WrongValue",
    "T2StaleData",
    "T3Empty404",
    "T4Timeout500",
    "T5SchemaDrift",
    "all_operators",
    "held_out_operators",
    "seen_operators",
]
