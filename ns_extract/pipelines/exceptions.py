"""Custom exceptions for pipeline operations."""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    pass


class InputError(PipelineError):
    """Raised when there are issues with pipeline inputs."""

    pass


class ValidationError(PipelineError):
    """Raised when output validation fails."""

    pass


class ProcessingError(PipelineError):
    """Raised when study processing fails."""

    def __init__(self, study_id: str, message: str):
        self.study_id = study_id
        super().__init__(f"Error processing study {study_id}: {message}")


class FileOperationError(PipelineError):
    """Raised when file operations (read/write) fail."""

    pass


class ConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""

    pass
