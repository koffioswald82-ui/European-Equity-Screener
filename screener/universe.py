import pandas as pd

try:
    import financedatabase as fd
    _FD_AVAILABLE = True
except ImportError:
    _FD_AVAILABLE = False

# Fallback: a curated sample of large-cap European tickers (yfinance symbols)
_FALLBACK_TICKERS = [
    # France (CAC 40 blue chips)
    "MC.PA", "TTE.PA", "SAN.PA", "OR.PA", "AIR.PA", "BNP.PA", "SU.PA",
    "AI.PA", "DG.PA", "RI.PA", "CS.PA", "CAP.PA", "DSY.PA", "KER.PA",
    # Germany (DAX blue chips)
    "SAP.DE", "SIE.DE", "ALV.DE", "MRK.DE", "DTE.DE", "BAYN.DE", "BMW.DE",
    "MBG.DE", "VOW3.DE", "EOAN.DE", "RWE.DE", "HEI.DE", "BAS.DE", "DB1.DE",
    # Netherlands
    "ASML.AS", "HEIA.AS", "INGA.AS", "PHIA.AS", "WKL.AS", "NN.AS", "ABN.AS",
    # Spain
    "IBE.MC", "SAN.MC", "BBVA.MC", "ITX.MC", "REP.MC", "TEF.MC",
    # Italy
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "TIT.MI", "ATL.MI",
    # Belgium / Switzerland listed in EUR
    "ABI.BR", "UCB.BR", "SOLB.BR",
]

EU_COUNTRIES = ["France", "Germany", "Netherlands", "Italy", "Spain", "Belgium",
                "Switzerland", "Sweden", "Denmark", "Finland", "Norway", "Austria",
                "Portugal", "Ireland", "Luxembourg"]


def get_universe(
    min_market_cap: float = 500e6,
    countries: list[str] | None = None,
    sectors: list[str] | None = None,
    use_financedatabase: bool = True,
) -> list[str]:
    """
    Return a list of yfinance ticker symbols for the investable European universe.

    Falls back to a hardcoded blue-chip list if financedatabase is not installed
    or returns no results.
    """
    if use_financedatabase and _FD_AVAILABLE:
        tickers = _fetch_from_fd(min_market_cap, countries or EU_COUNTRIES, sectors)
        if tickers:
            return tickers

    return _FALLBACK_TICKERS


def get_universe_df(
    min_market_cap: float = 500e6,
    countries: list[str] | None = None,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with ticker + metadata for the universe (requires financedatabase)."""
    if not _FD_AVAILABLE:
        return pd.DataFrame({"symbol": _FALLBACK_TICKERS})

    return _fetch_df_from_fd(min_market_cap, countries or EU_COUNTRIES, sectors)


# ── internal helpers ──────────────────────────────────────────────────────────

def _fetch_from_fd(min_market_cap: float, countries: list[str], sectors: list[str] | None) -> list[str]:
    try:
        df = _fetch_df_from_fd(min_market_cap, countries, sectors)
        return df["symbol"].dropna().tolist()
    except Exception:
        return []


def _fetch_df_from_fd(min_market_cap: float, countries: list[str], sectors: list[str] | None) -> pd.DataFrame:
    equities = fd.Equities()
    kwargs = {"country": countries, "currency": "EUR"}
    if sectors:
        kwargs["sector"] = sectors
    df = equities.select(**kwargs)
    df = df.reset_index().rename(columns={"index": "symbol"})

    if "market_cap" in df.columns:
        df = df[df["market_cap"] >= min_market_cap]

    # Keep only symbols that look like valid yfinance tickers
    df = df[df["symbol"].str.len() <= 12]
    return df.reset_index(drop=True)
