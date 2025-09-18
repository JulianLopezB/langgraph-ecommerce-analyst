import os, sys
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from agents.process_classifier import ProcessTypeClassifier
from domain.entities import ProcessType


def test_parse_classification_response_valid_json():
    classifier = ProcessTypeClassifier()
    response = json.dumps(
        {
            "process_type": "python",
            "confidence": 0.9,
            "reasoning": "Requires advanced analysis",
            "requires_aggregation": True,
            "complexity_level": "high",
            "suggested_tables": ["orders", "customers"],
        }
    )

    result = classifier._parse_classification_response(response, "How many customers?")

    assert result.process_type is ProcessType.PYTHON
    assert result.confidence == 0.9
    assert result.reasoning == "Requires advanced analysis"
    assert result.requires_aggregation is True
    assert result.complexity_level == "high"
    assert result.suggested_tables == ["orders", "customers"]


def test_parse_classification_response_unknown_process_type_defaults_to_sql(caplog):
    classifier = ProcessTypeClassifier()
    caplog.clear()
    response = json.dumps({"process_type": "unknown"})

    with caplog.at_level(logging.WARNING):
        result = classifier._parse_classification_response(response, "Some query")

    assert result.process_type is ProcessType.SQL
    assert "Unknown process type: unknown" in caplog.text


def test_parse_classification_response_non_json_ml_keywords():
    classifier = ProcessTypeClassifier()
    result = classifier._parse_classification_response(
        "Not a JSON response",
        "Can you predict future sales?",
    )

    assert result.process_type is ProcessType.PYTHON
    assert result.reasoning == "Fallback: Query contains ML/statistical keywords"
