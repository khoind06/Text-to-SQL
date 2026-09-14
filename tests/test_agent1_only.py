import pytest
from unittest.mock import MagicMock
from src.agents.orchestrator import FinancialOrchestrator
from src.pipelines.evaluation_pipeline import FinQAEvaluator

def test_orchestrator_sql_only_mode():
    mock_planner = MagicMock()
    mock_planner.run_node.return_value = {'sql_query': 'SELECT 100 AS result;', 'error': None}
    mock_risk = MagicMock()
    mock_risk.run_node.return_value = {'analysis_report': 'Báo cáo mẫu'}

    orchestrator = FinancialOrchestrator(
        db_path='data/database/finance.db',
        planner_agent=mock_planner,
        risk_agent=mock_risk
    )

    state_sql_only = orchestrator.run('What is the result?', sql_only=True)
    assert state_sql_only.get('sql_query') == 'SELECT 100 AS result;'
    assert state_sql_only.get('analysis_report') is None
    mock_risk.run_node.assert_not_called()

    state_full = orchestrator.run('What is the result?', sql_only=False)
    assert state_full.get('analysis_report') == 'Báo cáo mẫu'
    mock_risk.run_node.assert_called_once()

def test_evaluator_agent1_only_initialization():
    evaluator = FinQAEvaluator(
        db_path='data/database/finance.db',
        sample_size=3,
        agent1_only=True
    )
    assert evaluator.agent1_only is True
