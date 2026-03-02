"""
src/ingestion/data_loader.py
==============================
Load CSV, Parquet, and JSON sensor data and validate against the GNF
biomarker schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "schemas"
_BIOMARKER_SCHEMA_PATH = _SCHEMA_DIR / "biomarker_schema.json"


class DataLoader:
    """Load and validate sensor/biomarker data from multiple formats."""

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
        validate: bool = True,
    ) -> None:
        self.validate = validate
        self._schema: Optional[dict] = None

        if validate:
            sp = Path(schema_path) if schema_path else _BIOMARKER_SCHEMA_PATH
            if sp.exists():
                with open(sp, "r", encoding="utf-8") as fh:
                    self._schema = json.load(fh)
                logger.debug("Loaded biomarker schema from: %s", sp)
            else:
                logger.warning("Schema file not found at %s; validation disabled.", sp)
                self.validate = False

    def load(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load a file by auto-detecting its format."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return self.load_csv(p, **kwargs)
        elif suffix in (".parquet", ".pq"):
            return self.load_parquet(p, **kwargs)
        elif suffix == ".json":
            return self.load_json(p, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'.")

    def load_csv(self, path, parse_dates=None, **kwargs) -> pd.DataFrame:
        p = Path(path)
        logger.info("Loading CSV: %s", p)
        if parse_dates is None and "timestamp" in pd.read_csv(p, nrows=0).columns:
            parse_dates = ["timestamp"]
        df = pd.read_csv(p, parse_dates=parse_dates, **kwargs)
        self._post_load(df, source=str(p))
        return df

    def load_parquet(self, path, **kwargs) -> pd.DataFrame:
        p = Path(path)
        logger.info("Loading Parquet: %s", p)
        df = pd.read_parquet(p, **kwargs)
        self._post_load(df, source=str(p))
        return df

    def load_json(self, path, orient="records", **kwargs) -> pd.DataFrame:
        p = Path(path)
        logger.info("Loading JSON: %s", p)
        if self.validate and self._schema is not None:
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                self._validate_json_records(raw, source=str(p))
        df = pd.read_json(p, orient=orient, **kwargs)
        self._post_load(df, source=str(p))
        return df

    def load_batch(self, directory, pattern="**/*.csv", **kwargs) -> pd.DataFrame:
        d = Path(directory)
        files = sorted(d.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched '{pattern}' under '{d}'.")
        logger.info("Batch loading %d file(s) from: %s", len(files), d)
        dfs = []
        for f in files:
            try:
                df = self.load(f, **kwargs)
                df["source_file"] = f.name
                dfs.append(df)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)
        if not dfs:
            raise RuntimeError("All files in batch failed to load.")
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Batch loaded %d rows from %d file(s).", len(combined), len(dfs))
        return combined

    def _post_load(self, df: pd.DataFrame, source: str = "") -> None:
        n_rows, n_cols = df.shape
        logger.info("Loaded %d rows x %d columns from '%s'.", n_rows, n_cols, source)
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.warning("Could not parse 'timestamp' column: %s", exc)
        key_cols = ["subject_id", "session_id", "device_id", "biomarker_type"]
        for col in key_cols:
            if col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null > 0:
                    logger.warning("Column '%s' has %d null values.", col, n_null)

    def _validate_json_records(self, records: list[dict], source: str = "") -> None:
        try:
            import jsonschema
        except ImportError:
            logger.warning("jsonschema not installed; skipping record validation.")
            return
        errors_found = 0
        for i, record in enumerate(records):
            try:
                jsonschema.validate(record, self._schema)
            except jsonschema.ValidationError as exc:
                errors_found += 1
                if errors_found <= 5:
                    logger.warning("Record %d validation error in %s: %s", i, source, exc.message)
        if errors_found > 0:
            logger.warning("%d/%d records failed schema validation in '%s'.", errors_found, len(records), source)
        else:
            logger.debug("All %d records passed schema validation.", len(records))
