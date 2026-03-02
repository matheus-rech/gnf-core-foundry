"""
src/validation/schema_validator.py
=====================================
Validate Pandas DataFrames and JSON records against JSON Schemas.

Provides:
- JSON Schema validation for raw biomarker records
- DataFrame column-level validation (type, range, completeness)
- Custom rule registration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a schema validation run.

    Attributes:
        is_valid: True if no errors.
        errors: List of error message strings.
        warnings: List of warning strings.
        n_records_validated: Number of records checked.
        n_errors: Number of errors found.
        n_warnings: Number of warnings found.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_records_validated: int = 0
    n_errors: int = 0
    n_warnings: int = 0

    def __repr__(self) -> str:
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"errors={self.n_errors}, warnings={self.n_warnings}, "
            f"records={self.n_records_validated})"
        )

    def raise_if_invalid(self) -> None:
        """Raise ValueError if not valid.

        Raises:
            ValueError: With all error messages.
        """
        if not self.is_valid:
            raise ValueError(
                f"Validation failed with {self.n_errors} error(s):\n"
                + "\n".join(f"  - {e}" for e in self.errors[:10])
            )


class SchemaValidator:
    """Validate DataFrames and JSON records against GNF schemas.

    Args:
        schema: JSON Schema dict (optional).
        schema_path: Path to a JSON schema file (optional, takes precedence
            over ``schema`` if both are provided).
    """

    def __init__(
        self,
        schema: Optional[dict] = None,
        schema_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._schema: Optional[dict] = None
        self._custom_rules: list[Callable[[pd.DataFrame], list[str]]] = []

        if schema_path is not None:
            import json
            p = Path(schema_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    self._schema = json.load(fh)
            else:
                logger.warning("Schema path not found: %s", p)

        if self._schema is None and schema is not None:
            self._schema = schema

    def add_rule(self, rule_fn: Callable[[pd.DataFrame], list[str]]) -> None:
        """Add a custom validation rule function.

        Args:
            rule_fn: Callable(df) → list of error strings.  Return empty list
                if no errors.
        """
        self._custom_rules.append(rule_fn)

    def validate_json_records(self, records: list[dict]) -> ValidationResult:
        """Validate a list of JSON records against the loaded schema.

        Args:
            records: List of raw record dicts.

        Returns:
            ValidationResult.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if self._schema is None:
            warnings.append("No schema loaded; skipping JSON record validation.")
            return ValidationResult(
                is_valid=True, warnings=warnings,
                n_records_validated=len(records), n_warnings=1,
            )

        try:
            import jsonschema
        except ImportError:
            warnings.append("jsonschema not installed; skipping validation.")
            return ValidationResult(
                is_valid=True, warnings=warnings,
                n_records_validated=len(records), n_warnings=1,
            )

        for i, record in enumerate(records):
            try:
                jsonschema.validate(record, self._schema)
            except jsonschema.ValidationError as exc:
                errors.append(f"Record[{i}]: {exc.message}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            n_records_validated=len(records),
            n_errors=len(errors),
            n_warnings=len(warnings),
        )

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
        column_dtypes: Optional[dict[str, str]] = None,
        column_ranges: Optional[dict[str, tuple[float, float]]] = None,
        max_null_pct: float = 10.0,
    ) -> ValidationResult:
        """Validate a DataFrame for completeness, types, and ranges.

        Args:
            df: DataFrame to validate.
            required_columns: Column names that must be present.
            column_dtypes: Dict of column → expected dtype ('numeric', 'string',
                'datetime', 'bool').
            column_ranges: Dict of column → (min, max) expected value range.
            max_null_pct: Maximum allowed null percentage per column (0–100).

        Returns:
            ValidationResult.
        """
        errors: list[str] = []
        warnings: list[str] = []
        n = len(df)

        # Required columns
        for col in (required_columns or []):
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing.")

        # Dtype checks
        for col, expected_dtype in (column_dtypes or {}).items():
            if col not in df.columns:
                continue
            if expected_dtype == "numeric":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' expected numeric dtype; got {df[col].dtype}.")
            elif expected_dtype == "string":
                if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                    warnings.append(f"Column '{col}' expected string dtype; got {df[col].dtype}.")
            elif expected_dtype == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    warnings.append(f"Column '{col}' expected datetime dtype; got {df[col].dtype}.")

        # Null percentage checks
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            if null_pct > max_null_pct:
                warnings.append(
                    f"Column '{col}' has {null_pct:.1f}% null values "
                    f"(threshold: {max_null_pct}%)."
                )

        # Range checks
        for col, (rmin, rmax) in (column_ranges or {}).items():
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            out_of_range = ((df[col] < rmin) | (df[col] > rmax)).sum()
            if out_of_range > 0:
                errors.append(
                    f"Column '{col}' has {out_of_range} values outside "
                    f"range [{rmin}, {rmax}]."
                )

        # Custom rules
        for rule in self._custom_rules:
            try:
                rule_errors = rule(df)
                errors.extend(rule_errors)
            except Exception as exc:
                warnings.append(f"Custom rule raised exception: {exc}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            n_records_validated=n,
            n_errors=len(errors),
            n_warnings=len(warnings),
        )
