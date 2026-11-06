from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import streamlit as st

from common import get_bundle, page_setup


page_setup("Model governance", "📚")
bundle = get_bundle()
summary = bundle["training_summary"]
st.markdown(
    f"""
### Intended use

Education and portfolio demonstration of calibrated one-month PD, policy
decisioning, expected loss, explanations and monitoring.

### Prohibited use

Real credit decisions, pricing, adverse action, regulatory capital, or any
automated decision about a person.

### Model inventory

- Version: `{bundle['model_version']}`
- Champion: `{bundle['champion_name']}`
- Training rows: `{summary['rows']:,}`
- Test rows: `{summary['test_rows']:,}`
- Source period: 2005
- Target: default in the following month
- Validation: stratified random holdout (no dates exist in the source)
- Sensitive/demographic fields: excluded from model inputs and reason codes

### Core assumptions

- LGD: `{bundle['policy_config']['lgd']:.0%}`
- Interest rate scenario: `{bundle['policy_config']['annual_interest_rate']:.0%}`
- Operating cost scenario: `{bundle['policy_config']['operating_cost_rate']:.0%}`
- Currency: `{bundle['policy_config']['currency']}`

### Known limitations

The dataset is old, geography-specific, contains only existing customers, has
no rejected applicants, and lacks income, pricing, recoveries and event dates.
The financial policy and drift batch are simulations. Local reason-code
sensitivities are associative rather than causal.
"""
)
st.info("Full governance notes are in reports/model_card.md and reports/validation_report.md.")
