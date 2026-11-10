import pandas as pd
import pytest

from src.services.screening.models import HardFilterConfig, ScreeningConfig
from src.services.watchlist_digest import choose_candidates, parse_codes, parse_holdings, selection_summary, validate_daily_history


def test_snapshot_falls_back_when_first_source_has_no_watchlist_coverage(monkeypatch):
    from src.services import watchlist_digest as digest
    from src.services.screening import snapshot

    responses = {
        "sina": pd.DataFrame([{"code": "600111", "name": "Other"}]),
        "efinance": pd.DataFrame([{"code": "600499", "name": "A"}, {"code": "600000", "name": "B"}]),
    }
    calls = []
    def fetch(sources, **_kwargs):
        calls.append(sources[0])
        return responses[sources[0]].copy()
    monkeypatch.setattr(snapshot, "fetch_snapshot_with_fallback", fetch)
    result = digest.fetch_watchlist_snapshot({"600499", "600000"})
    assert set(result.code) == {"600499", "600000"}
    assert calls == ["sina", "efinance"]


def test_snapshot_uses_best_partial_coverage_when_all_sources_incomplete(monkeypatch):
    from src.services import watchlist_digest as digest
    from src.services.screening import snapshot

    def fetch(sources, **_kwargs):
        if sources[0] == "sina":
            return pd.DataFrame([{"code": "600499", "name": "A"}])
        if sources[0] == "efinance":
            return pd.DataFrame([{"code": "600499", "name": "A"}, {"code": "600000", "name": "B"}])
        raise RuntimeError("unavailable")
    monkeypatch.setattr(snapshot, "fetch_snapshot_with_fallback", fetch)
    assert len(digest.fetch_watchlist_snapshot({"600499", "600000", "601888"})) == 2
    with pytest.raises(ValueError, match="watchlist"):
        digest.fetch_watchlist_snapshot({"600123"})


def test_parse_codes_keeps_unique_mainland_codes_and_rejects_other_markets():
    assert parse_codes("600499,300750,688507,002724,600499,159516,CRCL") == [
        "600499", "002724"
    ]


def test_holdings_are_deduplicated_and_invalid_input_is_rejected():
    assert parse_holdings("002724,002724") == ["002724"]
    assert parse_holdings("NO_HOLDINGS") == []
    import pytest
    with pytest.raises(ValueError):
        parse_holdings("002724,invalid!")
    with pytest.raises(ValueError):
        parse_holdings("HK01888")


def test_raw_daily_history_rejects_missing_prices_and_negative_volume():
    import pytest
    history = pd.DataFrame({"日期": ["2026-11-04"], "开盘": [10], "最高": [11], "最低": [9], "收盘": [10], "成交量": [100]})
    assert len(validate_daily_history(history)) == 1
    with pytest.raises(ValueError):
        validate_daily_history(history.drop(columns=["开盘"]))
    with pytest.raises(ValueError):
        validate_daily_history(history.assign(成交量=-1))
    with pytest.raises(ValueError):
        validate_daily_history(pd.concat([history, history], ignore_index=True))
    with pytest.raises(ValueError):
        validate_daily_history(history.assign(最高=float("inf")))
    with pytest.raises(ValueError, match="missing sessions"):
        validate_daily_history(history, expected_sessions=pd.DatetimeIndex(["2026-11-03", "2026-11-04"]))


def test_choose_filters_watchlist_holdings_st_and_stale_before_ranking():
    rows = pd.DataFrame([
        {"code": "600499", "name": "A", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": ""},
        {"code": "002724", "name": "Holding", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": ""},
        {"code": "600000", "name": "ST Example", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": ""},
        {"code": "601001", "name": "Stale", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": "stale_cache"},
        {"code": "300750", "name": "Growth board", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": ""},
        {"code": "600999", "name": "Not in watchlist", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90, "daily_quality_flags": ""},
    ])
    rows["verified_fresh"] = True
    config = ScreeningConfig(hard_filters=HardFilterConfig(exclude_st=True))
    picks = choose_candidates(rows, ["600499", "002724", "600000", "601001", "300750"], ["002724"], config)
    assert picks == ["600499"]


def test_choose_at_most_five_deduplicated_and_missing_quality_never_passes():
    rows = pd.DataFrame([
        {"code": f"60000{i}", "name": f"N{i}", "price": 10, "amount": 200, "change_pct": i, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 90 if i != 6 else None, "daily_quality_flags": ""}
        for i in range(1, 8)
    ])
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    rows["verified_fresh"] = True
    config = ScreeningConfig(hard_filters=HardFilterConfig(exclude_st=True))
    picks = choose_candidates(rows, rows.code.tolist(), [], config)
    assert len(picks) == 5
    assert len(set(picks)) == 5
    assert "600006" not in picks


def test_unverified_freshness_never_qualifies():
    rows = pd.DataFrame([{"code": "600499", "name": "A", "price": 10, "amount": 200, "change_pct": 4, "turnover_rate": 4, "volume_ratio": 2, "daily_quality_score": 100, "daily_quality_flags": "", "verified_fresh": False}])
    assert choose_candidates(rows, ["600499"], [], ScreeningConfig()) == []


def test_real_volume_breakout_strategy_accepts_a_complete_qualifying_row():
    from pathlib import Path
    from src.services.screening.strategy import load_strategy

    strategy = load_strategy(Path("src/services/screening/strategies/volume_breakout.yaml"))
    row = {"code": "600499", "name": "A", "price": 10, "amount": 200_000_000, "change_pct": 4, "turnover_rate": 5, "volume_ratio": 3, "daily_quality_score": 90, "daily_quality_flags": "", "verified_fresh": True, "price_above_ma20": True, "signal_score": 70, "macd_status": "bullish", "breakout_20d_pct": 0, "range_20d_pct": 20, "volume_ratio_20d": 2, "body_pct": 1, "consolidation_days_20d": 9}
    assert choose_candidates(pd.DataFrame([row]), ["600499"], [], strategy.screening) == ["600499"]


def test_summary_distinguishes_candidates_from_holdings_and_failure():
    text = selection_summary({"candidates": [], "holdings": ["002724"], "holding_names": {"002724": "海洋王"}, "status": "screening_unavailable"})
    assert "002724 海洋王" in text
    assert "筛选不可用" in text
    assert "候选" in text


def test_summary_explicitly_shows_empty_holdings():
    text = selection_summary({"candidates": [], "holdings": [], "status": "no_qualifying_candidates"})
    assert "空仓" in text


def test_summary_includes_candidate_code_and_name():
    text = selection_summary({"candidates": [{"code": "600499", "name": "科达制造"}], "holdings": ["002724"], "status": "screened"})
    assert "600499 科达制造" in text
    assert "最近一个完整交易日" in text
    assert "不是实时行情" in text


def test_main_fails_closed_to_holdings_when_screening_raises(monkeypatch, tmp_path):
    import json
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499,002724,300750")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "002724")
    output = tmp_path / "result.json"
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(output))
    monkeypatch.setattr(digest, "is_cn_trading_day", lambda: True)
    monkeypatch.setattr(digest, "lookup_holding_names", lambda _: {})
    monkeypatch.setattr(digest, "select_watchlist", lambda *_: (_ for _ in ()).throw(RuntimeError("source unavailable")))
    monkeypatch.setattr(digest, "send_selection_summary", lambda _: True)
    assert digest.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["stocks"] == ["002724"]
    assert result["candidates"] == []
    assert result["status"] == "screening_unavailable"
    assert result["summary_sent"] is True


def test_summary_failure_is_recorded_without_discarding_holdings(monkeypatch, tmp_path):
    import json
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499,002724")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "002724")
    output = tmp_path / "result.json"
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(output))
    monkeypatch.setattr(digest, "is_cn_trading_day", lambda: True)
    monkeypatch.setattr(digest, "lookup_holding_names", lambda _: {})
    monkeypatch.setattr(digest, "select_watchlist", lambda *_: ([], False))
    monkeypatch.setattr(digest, "send_selection_summary", lambda _: False)
    assert digest.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary_sent"] is False
    assert result["stocks"] == ["002724"]


def test_non_trading_day_skips_without_sending_or_screening(monkeypatch, tmp_path):
    import json
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "002724")
    output = tmp_path / "result.json"
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(output))
    monkeypatch.setattr(digest, "is_cn_trading_day", lambda: False)
    monkeypatch.setattr(digest, "select_watchlist", lambda *_: (_ for _ in ()).throw(AssertionError("should not screen")))
    monkeypatch.setattr(digest, "send_selection_summary", lambda *_: (_ for _ in ()).throw(AssertionError("should not send")))
    assert digest.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["skip"] is True


def test_calendar_unavailable_alerts_and_fails_closed(monkeypatch, tmp_path):
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "002724")
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(tmp_path / "result.json"))
    monkeypatch.setattr(digest, "is_cn_trading_day", lambda: (_ for _ in ()).throw(ValueError("calendar unavailable")))
    sent = []
    monkeypatch.setattr(digest, "send_selection_summary", lambda result: sent.append(result) or True)
    assert digest.main() == 1
    assert sent[0]["status"] == "calendar_unavailable"


def test_missing_holdings_does_not_reuse_full_watchlist(monkeypatch, tmp_path):
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499,002724")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "")
    output = tmp_path / "result.json"
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(output))
    assert digest.main() == 1
    assert not output.exists()


def test_force_run_bypasses_calendar_for_manual_testing(monkeypatch, tmp_path):
    import json
    from src.services import watchlist_digest as digest

    monkeypatch.setenv("STOCK_LIST", "600499")
    monkeypatch.setenv("WATCHLIST_HOLDINGS", "002724")
    monkeypatch.setenv("WATCHLIST_FORCE_RUN", "true")
    output = tmp_path / "result.json"
    monkeypatch.setenv("WATCHLIST_SELECTION_OUTPUT", str(output))
    monkeypatch.setattr(digest, "is_cn_trading_day", lambda: False)
    monkeypatch.setattr(digest, "lookup_holding_names", lambda _: {})
    monkeypatch.setattr(digest, "select_watchlist", lambda *_: ([], False))
    monkeypatch.setattr(digest, "send_selection_summary", lambda _: True)
    assert digest.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["stocks"] == ["002724"]


def test_selection_checks_snapshot_against_history(monkeypatch):
    from types import SimpleNamespace
    from src.core import trading_calendar
    from src.services import watchlist_digest as digest
    from src.services.screening import daily, snapshot, strategy

    df = pd.DataFrame([{"code": "600499", "name": "A", "price": 888, "amount": 200, "change_pct": 999, "turnover_rate": 4, "volume_ratio": 2, "pe_ratio": -10}])
    monkeypatch.setattr(snapshot, "fetch_snapshot_with_fallback", lambda *_, **__: df.copy())
    monkeypatch.setattr(strategy, "load_strategy", lambda *_: SimpleNamespace(screening=ScreeningConfig()))
    monkeypatch.setattr(trading_calendar, "build_market_phase_context", lambda **_: SimpleNamespace(phase=trading_calendar.MarketPhase.PREMARKET))
    monkeypatch.setattr(trading_calendar, "get_trading_dates", lambda *_args: pd.DatetimeIndex(["2026-11-03", "2026-11-04"]))
    monkeypatch.setattr(daily, "daily_history_is_stale", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(daily, "fetch_daily_history", lambda *_args, **_kwargs: pd.DataFrame({"日期": ["2026-11-03", "2026-11-04"], "收盘": [10, 11], "开盘": [10, 10], "最高": [11, 11], "最低": [9, 9], "成交量": [100, 100]}))

    def enrich(frame, *, history_fetcher, **_kwargs):
        history_fetcher("600499")
        frame = frame.copy()
        frame["daily_quality_score"] = 100
        frame["daily_quality_flags"] = ""
        return frame

    monkeypatch.setattr(daily, "enrich_daily_features", enrich)
    def score_checked(frame, _config):
        assert frame.iloc[0]["price"] == 11
        assert abs(frame.iloc[0]["change_pct"] - 10) < 0.01
        assert pd.isna(frame.iloc[0]["amount"])
        assert "pe_ratio" not in frame.columns
        return frame.assign(screen_score=50)
    monkeypatch.setattr(digest, "compute_screen_scores", score_checked)
    assert digest.select_watchlist(["600499"], ["002724"]) == ([{"code": "600499", "name": "A"}], False)
    assert digest.select_watchlist(["600499", "600000"], ["002724"])[1] is True
    monkeypatch.setattr(daily, "daily_history_is_stale", lambda *_args, **_kwargs: True)
    import pytest
    with pytest.raises(ValueError, match="stale"):
        digest.select_watchlist(["600499"], ["002724"])
    monkeypatch.setattr(daily, "daily_history_is_stale", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trading_calendar, "build_market_phase_context", lambda **_: SimpleNamespace(phase=trading_calendar.MarketPhase.INTRADAY))
    with pytest.raises(ValueError, match="completed"):
        digest.select_watchlist(["600499"], ["002724"])


def test_workflow_requires_selection_before_analysis_and_does_not_fall_back():
    from pathlib import Path
    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8"))
    step = next(item for item in workflow["jobs"]["analyze"]["steps"] if item.get("name") == "执行股票分析")
    script = step["run"]
    assert "WATCHLIST_HOLDINGS" in step["env"]
    assert script.index("python -m src.services.watchlist_digest") < script.index("python main.py")
    assert 'get("skip", False)' in script
    assert '["stocks"]' in script
    assert 'get("summary_sent", False)' in script
    assert 'export STOCK_LIST="600519"' not in script
    assert '无候选且空仓' in script


def test_email_success_requires_email_channel(monkeypatch):
    from types import SimpleNamespace
    import sys
    from src.services import watchlist_digest as digest

    class Notification:
        def __init__(self, _config): pass
        def send_to_email(self, *_args, **_kwargs): return False

    monkeypatch.setitem(sys.modules, "src.config", SimpleNamespace(get_config=lambda: None))
    monkeypatch.setitem(sys.modules, "src.notification_sender.email_sender", SimpleNamespace(EmailSender=Notification))
    assert digest.send_selection_summary({"candidates": [], "holdings": ["002724"], "status": "screening_unavailable", "partial": False}) is False

    class SuccessfulNotification:
        def __init__(self, _config): pass
        def send_to_email(self, *_args, **_kwargs): return True

    monkeypatch.setitem(sys.modules, "src.notification_sender.email_sender", SimpleNamespace(EmailSender=SuccessfulNotification))
    assert digest.send_selection_summary({"candidates": [], "holdings": ["002724"], "status": "screening_unavailable", "partial": False}) is True
