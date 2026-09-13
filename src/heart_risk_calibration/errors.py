"""Exception types raised at package boundaries. Each message tells the user
what to do next."""
from __future__ import annotations


class HeartRiskError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(HeartRiskError):
    """A configuration file is missing, malformed, or internally inconsistent."""


class DataRootError(HeartRiskError):
    """DATA_ROOT is unset, does not exist, or lacks the expected data file."""


class SchemaError(HeartRiskError):
    """An input table lacks the expected columns or holds values outside the documented ranges."""


class TermsNotAcceptedError(HeartRiskError):
    """A download was requested without explicit acceptance of the data terms."""


class ValidationError(HeartRiskError):
    """A produced output file does not satisfy its schema."""
