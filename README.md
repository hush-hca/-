# Crypto Pure Alpha Long/Short Screener

Python tools for screening Binance USD-M futures altcoins by BTC/ETH beta-neutral residual returns.

## Files

- `index.html` - Vercel/static dashboard entrypoint.
- `api/run.py` - Vercel Python serverless API for live scans.
- `outputs/crypto_pure_alpha_screener.py` - CLI screener.
- `outputs/crypto_pure_alpha_dashboard.py` - local web dashboard.
- `outputs/requirements_pure_alpha_dashboard.txt` - Python dependencies.
- `requirements.txt` - Vercel deployment dependencies.

## Install

```powershell
pip install -r outputs\requirements_pure_alpha_dashboard.txt
```

## Run CLI

```powershell
python outputs\crypto_pure_alpha_screener.py --top-n 5 --lookback-hours 48
```

## Run Dashboard

```powershell
python outputs\crypto_pure_alpha_dashboard.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Deploy on Vercel

The repository includes `index.html`, `api/run.py`, `requirements.txt`, and `vercel.json`, so Vercel can serve the dashboard at `/` and the live screener API at `/api/run`.

## Notes

The model estimates per-asset BTC/ETH betas from recent hourly returns, then ranks the latest idiosyncratic residuals cross-sectionally. It uses public Binance Futures market data through `ccxt`; no API key is required for the included screener.
