# PREFLIGHT — Pine Script v6 Cockpit

Static validator for TradingView Pine Script v6. Single self-contained
`index.html` (no backend, no database) — `serve` just hosts the static file.

## Deploy to Railway (project already exists — link, don't create)

```bash
npm i -g @railway/cli      # if not already installed
railway login              # opens browser auth
railway link 0132eb33-6c8c-4963-9069-d7d1cf365ca8
railway up
```

`railway up` uploads this folder and deploys it using the start command in
`railway.json` (`serve -s . -l $PORT`, Railway injects `$PORT`).

## Run locally first (optional sanity check)

```bash
npm start
# open http://localhost:3000
```
