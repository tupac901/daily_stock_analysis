"""Select a bounded watchlist using the existing screening engine."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from src.services.screening.filter import apply_hard_filters, hard_filter_rejection_summary
from src.services.screening.models import ScreeningConfig
from src.services.screening.scorer import compute_screen_scores

logger = logging.getLogger(__name__)
MAIN_BOARD = re.compile(r"^(?:600|601|603|605|000|001|002|003)\d{3}$")
HOLDING_CODE = re.compile(r"^\d{6}$")


def parse_codes(value: str) -> list[str]:
    return list(dict.fromkeys(c for c in re.split(r"[,\s]+", value.upper()) if MAIN_BOARD.fullmatch(c)))


def parse_holdings(value: str) -> list[str]:
    if value.strip().upper() == "NO_HOLDINGS":
        return []
    codes = list(dict.fromkeys(c for c in re.split(r"[,\s]+", value.upper()) if c))
    if not codes or any(not HOLDING_CODE.fullmatch(code) for code in codes):
        raise ValueError("Holdings must contain six-digit mainland security codes")
    return codes


def validate_daily_history(hist: pd.DataFrame, *, expected_sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    raw = hist.rename(columns={"日期": "date", "trade_date": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}).copy()
    fields = ["open", "high", "low", "close", "volume"]
    if raw.empty or not {"date", *fields}.issubset(raw.columns):
        raise ValueError("daily OHLCV data is incomplete")
    dates = pd.to_datetime(raw["date"].astype(str).str[:10], format="mixed", errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("daily bar dates are invalid or duplicated")
    if expected_sessions is not None and set(pd.DatetimeIndex(dates).normalize()) != set(pd.DatetimeIndex(expected_sessions).normalize()):
        raise ValueError("daily history has missing sessions")
    values = raw[fields].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("daily OHLCV contains missing or non-finite values")
    if (values[["open", "high", "low", "close"]] <= 0).any().any() or (values["volume"] < 0).any():
        raise ValueError("daily OHLCV contains invalid prices or volume")
    if (values["high"] < values[["open", "low", "close"]].max(axis=1)).any() or (values["low"] > values[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("daily OHLC prices are inconsistent")
    return raw


def reliable_daily_mask(df: pd.DataFrame) -> pd.Series:
    quality = pd.to_numeric(df["daily_quality_score"], errors="coerce")
    flags = df["daily_quality_flags"].fillna("").astype(str)
    return quality.between(80, 100) & df["verified_fresh"].eq(True) & ~flags.str.contains(r"stale|fetch_failed|invalid_ohlc|non_positive_price|negative_volume|missing_", case=False)


def summarize_history_failures(errors: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for error in errors:
        message = str(error).partition(":")[2].strip().lower()
        if "missing sessions" in message:
            category = "missing_sessions"
        elif "stale or its trading date" in message:
            category = "stale_or_unknown_date"
        elif any(term in message for term in ("invalid", "incomplete", "missing", "inconsistent", "no dates", "future dates")):
            category = "invalid_data"
        else:
            category = "other_or_provider_failure"
        counts[category] += 1
    return dict(counts)


def fetch_verified_history(code, fetch, verify, kwargs):
    frame = fetch(code, **kwargs)
    try:
        return verify(frame)
    except ValueError as exc:
        if str(exc) != "daily history has missing sessions" or kwargs.get("source") != "auto":
            raise
        original_error = exc
    first_source = frame.attrs.get("daily_source")
    for source in ("tencent", "sina", "akshare", "baostock"):
        if source == first_source:
            continue
        try:
            alternative = fetch(code, **{**kwargs, "source": source, "retries": 0})
            return verify(alternative)
        except Exception:
            continue
    raise original_error


def choose_candidates(rows: pd.DataFrame, watchlist: list[str], holdings: list[str], config: ScreeningConfig) -> list[str]:
    if rows.empty or not {"code", "name", "daily_quality_score", "daily_quality_flags", "verified_fresh"}.issubset(rows.columns):
        return []
    df = rows.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].isin(parse_codes(",".join(watchlist))) & ~df["code"].isin(holdings)]
    df = df[df["name"].notna() & df["name"].astype(str).str.strip().ne("")]
    df = df[~df["name"].astype(str).str.contains(r"ST|退", case=False, na=True)]
    df = df[reliable_daily_mask(df)]
    logger.info("Watchlist screening: reliable eligible rows=%d", len(df))
    if df.empty:
        return []
    logger.info("Watchlist screening filter counts: %s", "; ".join(hard_filter_rejection_summary(df, config.hard_filters, limit=30)) or "none rejected")
    df = apply_hard_filters(df, config.hard_filters).drop_duplicates(subset="code")
    logger.info("Watchlist screening: qualifying rows=%d", len(df))
    if df.empty:
        return []
    scored = compute_screen_scores(df, config).sort_values(["screen_score", "code"], ascending=[False, True])
    return scored["code"].head(5).tolist()


def fetch_watchlist_snapshot(universe: set[str]) -> pd.DataFrame:
    from src.services.screening.snapshot import fetch_snapshot_with_fallback

    best = None
    best_count = 0
    for source in ("sina", "efinance", "akshare_em", "em_datacenter"):
        try:
            frame = fetch_snapshot_with_fallback([source], required_columns=["code", "name"]).copy()
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            named = frame["name"].notna() & frame["name"].astype(str).str.strip().ne("")
            count = frame.loc[named & frame["code"].isin(universe), "code"].nunique()
            logger.info("Snapshot %s covers %d of %d watchlist stocks", source, count, len(universe))
            if count > best_count:
                best, best_count = frame, count
            if count == len(universe):
                break
        except Exception as exc:
            logger.warning("Snapshot %s unavailable: %s", source, exc)
    if best is None:
        raise ValueError("watchlist is absent from all available market snapshots")
    return best


def select_watchlist(watchlist: list[str], holdings: list[str]) -> tuple[list[dict[str, str]], bool]:
    from src.core.trading_calendar import MarketPhase, build_market_phase_context, get_trading_dates
    from src.services.screening.daily import _DAILY_FEATURE_DEFAULTS, _normalize_daily_history, daily_history_is_stale, enrich_daily_features, fetch_daily_history
    from src.services.screening.strategy import load_strategy

    strategy = load_strategy(Path(__file__).parent / "screening" / "strategies" / "volume_breakout.yaml")
    universe = set(watchlist) - set(holdings)
    if not universe:
        return [], False
    snapshot = fetch_watchlist_snapshot(universe)
    snapshot["code"] = snapshot["code"].astype(str).str.zfill(6)
    snapshot = snapshot[snapshot["code"].isin(universe)].copy()
    seen_codes = set(snapshot["code"])
    if not seen_codes:
        raise ValueError("watchlist is absent from the market snapshot")
    named = snapshot["name"].notna() & snapshot["name"].astype(str).str.strip().ne("")
    missing_names = not bool(named.all())
    if not named.any():
        raise ValueError("watchlist names are unavailable")
    snapshot = snapshot[named & ~snapshot["name"].astype(str).str.contains(r"ST|退", case=False, na=True)]
    snapshot = snapshot.drop_duplicates(subset="code")
    if snapshot.empty:
        return [], missing_names or len(seen_codes) < len(universe)

    daily_prices: dict[str, tuple[float, float]] = {}

    def checked_history(code: str, **kwargs: object) -> pd.DataFrame:
        return fetch_verified_history(code, fetch_daily_history, lambda hist: validate_completed_history(code, hist), kwargs)

    def validate_completed_history(code: str, hist: pd.DataFrame) -> pd.DataFrame:
        context = build_market_phase_context(market="cn")
        if context.phase == MarketPhase.UNKNOWN:
            raise ValueError("daily history is stale or its trading date is unavailable")
        date_column = next((name for name in ("date", "日期", "trade_date") if name in hist.columns), None)
        if date_column is None:
            raise ValueError("daily history has no dates")
        dates = pd.to_datetime(hist[date_column].astype(str).str[:10], format="mixed", errors="coerce")
        if dates.isna().any() or (dates.dt.date > context.session_date).any():
            raise ValueError("daily history contains invalid or future dates")
        incomplete_session = context.phase in (MarketPhase.PREMARKET, MarketPhase.INTRADAY, MarketPhase.LUNCH_BREAK, MarketPhase.CLOSING_AUCTION)
        if context.effective_daily_bar_date > context.session_date or (incomplete_session and context.effective_daily_bar_date >= context.session_date):
            raise ValueError("completed trading date is invalid")
        hist = hist.loc[dates.dt.date <= context.effective_daily_bar_date].copy()
        if hist.empty or dates.loc[hist.index].max().date() != context.effective_daily_bar_date or daily_history_is_stale(hist, code=code):
            raise ValueError("daily history is stale or its trading date is unavailable")
        raw = validate_daily_history(hist)
        dates = pd.to_datetime(raw["date"].astype(str).str[:10], format="mixed", errors="coerce")
        sessions = get_trading_dates("cn", dates.min().date(), dates.max().date())
        if sessions is None:
            raise ValueError("daily trading calendar is unavailable")
        normalized = _normalize_daily_history(validate_daily_history(hist, expected_sessions=sessions))
        close = pd.to_numeric(normalized["close"], errors="coerce").dropna()
        if len(close) < 2 or close.iloc[-1] <= 0 or close.iloc[-2] <= 0:
            raise ValueError("daily close is missing or invalid")
        daily_prices[code] = (float(close.iloc[-1]), float((close.iloc[-1] / close.iloc[-2] - 1) * 100))
        return hist

    enriched = enrich_daily_features(snapshot, max_rows=len(snapshot), source="auto", fetch_retries=1, max_workers=3, history_fetcher=checked_history)
    logger.info("Watchlist history failure categories: %s", summarize_history_failures(enriched.attrs.get("daily_errors", [])))
    logger.info("Watchlist history successful sources: %s", enriched.attrs.get("daily_source_counts", {}))
    if not daily_prices:
        raise ValueError("no watchlist daily histories could be verified")
    enriched["verified_fresh"] = enriched["code"].isin(daily_prices)
    enriched["price"] = enriched["code"].map(lambda code: daily_prices.get(code, (None, None))[0])
    enriched["change_pct"] = enriched["code"].map(lambda code: daily_prices.get(code, (None, None))[1])
    enriched["amount"] = pd.NA
    enriched["turnover_rate"] = pd.NA
    enriched["volume_ratio"] = enriched.get("volume_ratio_20d", pd.NA)
    allowed = set(_DAILY_FEATURE_DEFAULTS) | {"code", "name", "verified_fresh", "price", "change_pct", "amount", "turnover_rate", "volume_ratio"}
    enriched = enriched.loc[:, [column for column in enriched.columns if column in allowed]]
    reliable_count = int(reliable_daily_mask(enriched).sum())
    logger.info("Watchlist history verification: snapshot=%d, validated=%d, reliable=%d", len(snapshot), len(daily_prices), reliable_count)
    if reliable_count == 0:
        raise ValueError("no complete, reliable daily features")
    # Snapshot-only filters cannot be trusted before the open without a provider timestamp.
    daily_filters = replace(strategy.screening.hard_filters, amount_min=None, turnover_rate_min=None, volume_ratio_min=None)
    config = replace(strategy.screening, hard_filters=daily_filters)
    codes = choose_candidates(enriched, watchlist, holdings, config)
    names = dict(zip(enriched["code"], enriched["name"].astype(str)))
    return [{"code": code, "name": names[code]} for code in codes], missing_names or reliable_count < len(snapshot) or len(seen_codes) < len(universe)


def selection_summary(result: dict) -> str:
    status_text = {
        "screened": "通过项目内置放量突破规则筛选，候选不代表买入建议。",
        "no_qualifying_candidates": "本次没有符合条件的自选股，不强行凑满五只。",
        "screening_unavailable": "筛选不可用：行情或日线数据无法验证，本次不提供候选。",
        "calendar_unavailable": "交易日历不可用：本次未筛选或分析，请检查配置和日历更新。",
    }
    return "\n".join([
        "# 自选筛选与持仓",
        status_text[result["status"]],
        "## 候选（最多五只）",
        ", ".join(f'{item["code"]} {item["name"]}' for item in result["candidates"]) or "无",
        "## 持仓（单独分析）",
        ", ".join(f'{code} {result.get("holding_names", {}).get(code, "")}'.strip() for code in result["holdings"]) or "空仓",
        "筛选基于最近一个完整交易日的日线，不是实时行情或当日竞价。成交额、换手率与 ST 状态仍需核验。",
        "部分自选日线不可用，候选覆盖不完整。" if result.get("partial") else "",
        "持仓与自选来自配置快照，不会自动同步同花顺。数据可能延迟，新闻可能缺失；请核对后自行决策。",
    ])


def send_selection_summary(result: dict) -> bool:
    from src.config import get_config
    from src.notification_sender.email_sender import EmailSender

    try:
        return bool(EmailSender(get_config()).send_to_email(selection_summary(result)))
    except Exception:
        logger.exception("Selection summary notification failed")
        return False


def lookup_holding_names(holdings: list[str]) -> dict[str, str]:
    try:
        from data_provider.tencent_fetcher import TencentFetcher

        fetcher = TencentFetcher()
        return {code: name for code in holdings if (name := fetcher.get_stock_name(code))}
    except Exception:
        logger.warning("Unable to resolve holdings names", exc_info=True)
        return {}


def is_cn_trading_day() -> bool:
    from src.core.trading_calendar import get_trading_dates

    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    dates = get_trading_dates("cn", today, today)
    if dates is None:
        raise ValueError("Chinese trading calendar unavailable")
    return len(dates) > 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        holdings = parse_holdings(os.environ.get("WATCHLIST_HOLDINGS", ""))
        watchlist = parse_codes(os.environ.get("STOCK_LIST", ""))
        if not watchlist:
            raise ValueError("Watchlist has no mainland main-board stocks")
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    try:
        force_run = os.environ.get("WATCHLIST_FORCE_RUN", "").lower() == "true"
        if not force_run and not is_cn_trading_day():
            Path(os.environ["WATCHLIST_SELECTION_OUTPUT"]).write_text(json.dumps({"skip": True}), encoding="utf-8")
            return 0
    except ValueError as exc:
        logger.error("Trading calendar error: %s", exc)
        send_selection_summary({"candidates": [], "holdings": holdings, "status": "calendar_unavailable"})
        return 1
    try:
        picks, partial = select_watchlist(watchlist, holdings)
        status = "screened" if picks else "no_qualifying_candidates"
    except Exception as exc:
        logger.warning("Screening unavailable; holdings only: %s", exc)
        picks = []
        partial = True
        status = "screening_unavailable"
    output = {"candidates": picks, "holdings": holdings, "holding_names": lookup_holding_names(holdings), "status": status, "partial": partial, "stocks": list(dict.fromkeys([item["code"] for item in picks] + holdings))}
    output["summary_sent"] = send_selection_summary(output)
    Path(os.environ["WATCHLIST_SELECTION_OUTPUT"]).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    if not output["summary_sent"]:
        logger.error("Selection summary was not delivered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
