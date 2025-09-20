"""Application factory for creating fully wired controllers and services."""

from __future__ import annotations

from infrastructure.config import get_config


def _create_config():
    """Create system configuration instance."""
    return get_config()


def _bind_services(config):
    """Instantiate and bind service implementations."""
    import infrastructure.execution as execution_module
    import infrastructure.llm as llm_module
    import infrastructure.persistence as persistence_module
    from infrastructure.execution.secure_executor import SecureExecutor
    from infrastructure.execution.validator import CodeValidator
    from infrastructure.llm.gemini import GeminiClient
    from infrastructure.persistence.bigquery import BigQueryRepository

    llm_client = GeminiClient(api_key=config.api_configurations.gemini_api_key)
    data_repository = BigQueryRepository(
        project_id=config.api_configurations.bigquery_project_id,
        dataset_id=config.api_configurations.dataset_id,
    )
    # Enable tracing in development environment
    enable_tracing = config.environment == "development"
    secure_executor = SecureExecutor(
        config.execution_limits, 
        enable_tracing=enable_tracing,
        enable_detailed_tracing=False  # Can be enabled for deep debugging
    )
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

    from application.controllers import AnalysisController
    from workflow import AnalysisWorkflow

    workflow = AnalysisWorkflow()
    return AnalysisController(workflow=workflow)
