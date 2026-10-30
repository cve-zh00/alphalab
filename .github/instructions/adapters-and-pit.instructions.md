---
applyTo: "src/alphalab/adapters/**/*.py"
---

# Adapters, Point-in-Time Storage & External I/O Instructions

When editing files under `src/alphalab/adapters/`:

1. **Point-in-Time (PIT) Anti-Leakage:**
   - Every read method on `ParquetPITStore`, `EdgarClient`, or news/filing adapters must require an explicit `knowledge_date` and enforce `observed_at <= knowledge_date`.
   - Never expose an unfiltered query method that returns future filings or restatements.

2. **Parameterized SQL Only:**
   - All DuckDB (`pit_store.py`) and SQLite (`journal_sqlite.py`, `execution_log.py`, `abstention_store.py`, `ledger_sqlite.py`) queries must use `?` bound parameters. Never interpolate external strings into SQL.

3. **Resilience, Rate Limiting & Secrets:**
   - Wrap HTTP calls (`alpaca_bars.py`, `alpaca_assets.py`, `alpaca_news.py`, `edgar.py`, `cik.py`) with `RateLimiter` and exponential backoff on retryable status codes (`429`, `500`, `502`, `503`, `504`) and transient transport errors.
   - Store credentials as `pydantic.SecretStr` in `AlpacaSettings` and unwrap `.get_secret_value()` only inside `auth_headers()`.
   - Preserve the dual live-trading lock in `AlpacaSettings`: routing to live Alpaca requires both `ALPHALAB_ALPACA_ENDPOINT=live` and `acknowledge_live=True`.
