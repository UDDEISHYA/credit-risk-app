"""Data loading and explicit data-quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT


TARGET = "DEFAULT_NEXT_MONTH"
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
MODEL_INPUT_COLUMNS = ["LIMIT_BAL", *PAY_STATUS_COLUMNS, *BILL_COLUMNS, *PAYMENT_COLUMNS]
REQUIRED_COLUMNS = ["ID", *MODEL_INPUT_COLUMNS, *DEMOGRAPHIC_COLUMNS, TARGET]


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    duplicate_ids: int
    invalid_target: int
    non_positive_limits: int
    invalid_age: int
    negative_payments: int
    invalid_pay_status: int
    unknown_sex: int
    unknown_education: int
    unknown_marriage: int
    missing_columns: list[str]

    @property
    def passed(self) -> bool:
        ignored_source_category_issues = (
            self.unknown_education + self.unknown_marriage
        )
        critical = (
            self.missing_cells
            + self.duplicate_ids
            + self.invalid_target
            + self.non_positive_limits
            + self.invalid_age
            + self.negative_payments
            + self.invalid_pay_status
            + self.unknown_sex
        )
        return not self.missing_columns and critical == 0 and ignored_source_category_issues >= 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def load_credit_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the normalized CSV or the original UCI `.xls` workbook."""
    path = Path(path or PROJECT_ROOT / "data/credit_card_clients.csv")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix.lower() == ".xls":
        frame = pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd")
        frame = frame.rename(columns={"default payment next month": TARGET})
    else:
        frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def validate_credit_data(frame: pd.DataFrame, strict: bool = False) -> DataQualityReport:
    """Validate structural, range and category expectations.

    Education values 0, 5 and 6 and marriage value 0 are known undocumented
    source categories. They are reported but are not critical because those
    fields are excluded from the underwriting model.
    """
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        report = DataQualityReport(
            rows=len(frame), columns=len(frame.columns), missing_cells=int(frame.isna().sum().sum()),
            duplicate_rows=int(frame.duplicated().sum()), duplicate_ids=0, invalid_target=0,
            non_positive_limits=0, invalid_age=0, negative_payments=0, invalid_pay_status=0,
            unknown_sex=0, unknown_education=0, unknown_marriage=0,
            missing_columns=missing_columns,
        )
        if strict:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return report

    report = DataQualityReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_cells=int(frame[REQUIRED_COLUMNS].isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_ids=int(frame["ID"].duplicated().sum()),
        invalid_target=int((~frame[TARGET].isin([0, 1])).sum()),
        non_positive_limits=int((frame["LIMIT_BAL"] <= 0).sum()),
        invalid_age=int((~frame["AGE"].between(18, 100)).sum()),
        negative_payments=int((frame[PAYMENT_COLUMNS] < 0).sum().sum()),
        invalid_pay_status=int(((frame[PAY_STATUS_COLUMNS] < -2) | (frame[PAY_STATUS_COLUMNS] > 8)).sum().sum()),
        unknown_sex=int((~frame["SEX"].isin([1, 2])).sum()),
        unknown_education=int((~frame["EDUCATION"].isin([1, 2, 3, 4])).sum()),
        unknown_marriage=int((~frame["MARRIAGE"].isin([1, 2, 3])).sum()),
        missing_columns=[],
    )
    if strict and not report.passed:
        raise ValueError(f"Critical data-quality checks failed: {report.to_dict()}")
    return report

