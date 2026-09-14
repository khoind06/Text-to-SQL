"""
sql_generator.py (Agent 1 Module Alias)
Re-exports the core NL2SQLGenerator and helper functions from src.text_to_sql.sql_generator.
"""

from src.text_to_sql.sql_generator import (
    NL2SQLGenerator,
    RiskGuardrails,
    SQLEngine,
    SecurityViolationError,
)

__all__ = [
    "NL2SQLGenerator",
    "RiskGuardrails",
    "SQLEngine",
    "SecurityViolationError",
]
