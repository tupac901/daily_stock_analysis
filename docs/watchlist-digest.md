# Watchlist Digest

The customized workflow is triggered externally at 08:30 Asia/Shanghai and
checks the A-share calendar before sending. Configure repository secrets:

- `STOCK_LIST`: the watchlist snapshot, comma-separated.
- `WATCHLIST_HOLDINGS`: the current holdings snapshot, comma-separated, six-digit
  mainland security codes only; required. Use `NO_HOLDINGS` for an explicitly
  empty portfolio. Missing configuration and other markets fail closed.
- `EMAIL_SENDER`, `EMAIL_PASSWORD`, and a supported model API configuration.

This configuration does not connect to a brokerage or synchronize positions or
watchlists. Update both snapshots whenever they change. The public repository's
workflow logs and artifacts can expose stock codes and reports.

The digest first restricts the pool to Shanghai and Shenzhen main-board stocks,
excludes names marked ST and existing holdings, and adapts the existing
`volume_breakout` screening strategy to the latest verified completed daily
bars. The snapshot supplies names only; without a trustworthy snapshot time,
intraday turnover, amount and volume-ratio filters are not used. Verify ST,
liquidity, suspension, news and current quotes separately before trading.
Up to five qualifying candidates and all configured holdings receive detailed
analysis. If screening fails, the digest reports the failure and analyzes only
the holdings. With no candidates and no holdings, only the summary is sent.
If the calendar cannot be validated, the job fails closed.

The selection summary is sent by email and the detailed report uses the configured
notification channels. They are separate notifications. Results
are research candidates, not orders or guarantees; the data and news sources may
be incomplete or delayed. GitHub schedules and data sources are best effort,
and reports can arrive after the configured start time.
