"""Application factory for creating fully wired controllers and services."""

from __future__ import annotations

from infrastructure.config import get_config


def _create_config():
    """Create system configuration instance."""
    return get_config()


def _bind_services(config):
    """Instantiate and bind service implementations."""
    from infrastructure.llm.gemini import GeminiClient
    from infrastructure.persistence.bigquery import BigQueryRepository
    from infrastructure.execution.executor import SecureExecutor
    from infrastructure.execution.validator import CodeValidator
    import infrastructure.llm as llm_module
    import infrastructure.persistence as persistence_module
    import infrastructure.execution as execution_module

    llm_client = GeminiClient(api_key=config.api_configurations.gemini_api_key)
    data_repository = BigQueryRepository(
        project_id=config.api_configurations.bigquery_project_id,
        dataset_id=config.api_configurations.dataset_id,
    )
    secure_executor = SecureExecutor(config.execution_limits)
    validator = CodeValidator(config.security_settings)

    llm_module.llm_client = llm_client
    persistence_module.data_repository = data_repository
    execution_module.secure_executor = secure_executor
    execution_module.validator = validator

    return {
        "llm_client": llm_client,
        "data_repository": data_repository,
        "secure_executor": secure_executor,
        "validator": validator,
    }


def create_analysis_controller():
    """Create a fully wired :class:`AnalysisController`."""
    config = _create_config()
    _bind_services(config)

    from workflow import AnalysisWorkflow
    from application.controllers import AnalysisController

    workflow = AnalysisWorkflow()
    return AnalysisController(workflow=workflow)
