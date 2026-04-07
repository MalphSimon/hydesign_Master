"""Bankability metrics helpers for yearly evaluation outputs."""

import numpy as np
import pandas as pd


DEFAULT_BANKABILITY_ASSUMPTIONS = {
    "debt_fraction": 0.70,
    "debt_interest_rate": 0.06,
    "debt_tenor_years": 20,
    "target_min_dscr": 1.20,
    "cfads_degradation_rate": None,
    "llcr_discount_rate": None,
}


def _to_float(value, default=np.nan):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _annuity_factor(interest_rate, years):
    if years <= 0:
        return np.nan
    if interest_rate <= 0:
        return 1.0 / years

    growth = (1.0 + interest_rate) ** years
    return (interest_rate * growth) / (growth - 1.0)


def _pv_factor(interest_rate, years):
    if years <= 0:
        return np.nan
    if interest_rate <= 0:
        return float(years)
    return (1.0 - (1.0 + interest_rate) ** (-years)) / interest_rate


def _infer_cfads_degradation_rate(row, cfg):
    user_rate = cfg.get("cfads_degradation_rate")
    if user_rate is not None:
        rate = _to_float(user_rate, default=0.0)
        if pd.notna(rate):
            return float(max(0.0, rate))

    return 0.0


def _npv_of_series(values, discount_rate):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    if discount_rate <= 0:
        return float(np.nansum(values))
    discounts = (1.0 + discount_rate) ** np.arange(1, values.size + 1)
    return float(np.nansum(values / discounts))


def calculate_bankability_metrics(row, assumptions=None):
    """Calculate DSCR/LLCR-style metrics from a yearly evaluation row.

    Parameters
    ----------
    row : dict-like
        Row containing at least CAPEX, revenues and OPEX values.
    assumptions : dict, optional
        Optional overrides for debt assumptions.

    Returns
    -------
    dict
        Bankability metrics ready to append to evaluation output.
    """
    cfg = dict(DEFAULT_BANKABILITY_ASSUMPTIONS)
    if assumptions:
        cfg.update(assumptions)

    capex = _to_float(row.get("CAPEX [MEuro]"))
    revenues = _to_float(row.get("Revenues [MEuro]"))
    opex = _to_float(row.get("OPEX [MEuro]"))

    debt_fraction = _to_float(cfg.get("debt_fraction"), default=0.70)
    debt_interest_rate = _to_float(cfg.get("debt_interest_rate"), default=0.06)
    debt_tenor_years = int(_to_float(cfg.get("debt_tenor_years"), default=20))
    target_min_dscr = _to_float(cfg.get("target_min_dscr"), default=1.20)
    llcr_discount_rate = _to_float(
        cfg.get("llcr_discount_rate"),
        default=debt_interest_rate,
    )
    cfads_degradation_rate = _infer_cfads_degradation_rate(row, cfg)

    debt_amount = capex * debt_fraction
    cfads = revenues - opex

    annuity_factor = _annuity_factor(debt_interest_rate, debt_tenor_years)
    if (
        pd.notna(annuity_factor)
        and annuity_factor > 0
        and pd.notna(debt_amount)
    ):
        annual_debt_service = debt_amount * annuity_factor
    else:
        annual_debt_service = np.nan

    if pd.notna(annual_debt_service) and annual_debt_service > 0:
        if pd.notna(cfads) and debt_tenor_years > 0:
            cfads_series = cfads * (1.0 - cfads_degradation_rate) ** np.arange(
                debt_tenor_years
            )
            dscr_series = cfads_series / annual_debt_service
            dscr = float(np.nanmin(dscr_series))
            dscr_avg = float(np.nanmean(dscr_series))
        else:
            dscr = np.nan
            dscr_avg = np.nan
    else:
        dscr = np.nan
        dscr_avg = np.nan

    if pd.notna(cfads) and debt_tenor_years > 0:
        cfads_series = cfads * (1.0 - cfads_degradation_rate) ** np.arange(
            debt_tenor_years
        )
        pv_cfads = _npv_of_series(cfads_series, llcr_discount_rate)
    else:
        pv_cfads = np.nan

    if pd.notna(debt_amount) and debt_amount > 0 and pd.notna(pv_cfads):
        llcr = pv_cfads / debt_amount
    else:
        llcr = np.nan

    if pd.notna(annual_debt_service) and annual_debt_service > 0 and pd.notna(
        target_min_dscr
    ) and pd.notna(cfads) and debt_tenor_years > 0:
        cfads_series = cfads * (1.0 - cfads_degradation_rate) ** np.arange(
            debt_tenor_years
        )
        dscr_series = cfads_series / annual_debt_service
        dscr_breach_years = int(np.sum(dscr_series < target_min_dscr))
    else:
        dscr_breach_years = np.nan

    if pd.notna(cfads) and pd.notna(annuity_factor) and annuity_factor > 0:
        if pd.notna(target_min_dscr) and target_min_dscr > 0:
            max_annual_debt_service = cfads / target_min_dscr
            max_debt_capacity = max_annual_debt_service / annuity_factor
        else:
            max_debt_capacity = np.nan
    else:
        max_debt_capacity = np.nan

    if pd.notna(max_debt_capacity) and pd.notna(debt_amount):
        debt_headroom = max_debt_capacity - debt_amount
    else:
        debt_headroom = np.nan

    return {
        "CFADS [MEuro/yr]": cfads,
        "CFADS degradation rate [-/yr]": cfads_degradation_rate,
        "Debt Amount [MEuro]": debt_amount,
        "Annual Debt Service [MEuro/yr]": annual_debt_service,
        "DSCR [-]": dscr,
        "DSCR Avg [-]": dscr_avg,
        "LLCR [-]": llcr,
        "DSCR Breach Years": dscr_breach_years,
        "Debt Headroom [MEuro]": debt_headroom,
        "Debt Headroom [% of CAPEX]": (
            debt_headroom / capex if pd.notna(capex) and capex != 0 else np.nan
        ),
    }
