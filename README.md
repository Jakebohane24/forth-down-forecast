# Fourth Down Forecast

An end-to-end NFL forecasting product that turns play-level data into
versioned weekly predictions through a two-stage XGBoost model, a FastAPI
service, PostgreSQL persistence, and a Next.js/TypeScript interface.

> The betting indicators are experimental analysis, not financial advice.

## What is included

- Leakage-conscious SQLite feature pipeline built from nflverse play-by-play
- Stage-one component models and stage-two home/away point models
- Expanding-window historical evaluation with all results retained
- Production artifact trained on completed 2018–2025 data
- Official model excludes wind so historical and live feature definitions match
- Pregame feature builder verified against historical training rows
- Seeded score simulation with split-tie probabilities and a frozen 60%
  confidence / −300 moneyline signal
- Immutable Parquet and relational prediction snapshots
- Separately stored final scores, closing moneylines, and signal settlement
- FastAPI with generated OpenAPI documentation
- Responsive Next.js 16, React 19, TypeScript, and Tailwind interface
- PostgreSQL production persistence with a SQLite development fallback
- Docker Compose and GitHub Actions CI

The four-point spread experiment is retained in research reports but is not
shown as a product recommendation because the larger rolling backtest found it
unreliable.

## Architecture

```text
nflverse schedule + play-by-play       market odds
                 \       Open-Meteo wind /
                  pregame feature builder
                           |
                 versioned XGBoost model
                           |
                  seeded score simulation
                           |
              immutable prediction snapshot
                           |
                       PostgreSQL
                           |
                        FastAPI
                           |
                  Next.js public dashboard
```

Training is offline. The public application reads stored prediction snapshots;
it never retrains a model during a web request.

Completed games join the immutable prediction to a separate final-result row.
The interface shows predicted versus actual scores, the captured home and away
moneylines, pick accuracy, and flat one-unit signal profit or loss.

## Model policy

The production model uses all completed data from 2018 through 2025. Its
features, architecture, and combined 60% confidence / −300 moneyline rule are frozen
before prospective 2026 evaluation.

The current feature definition requires five completed games from each team in
the same season. Therefore Version 1 intentionally does not publish predictions
for early-season games without sufficient history. Carrying prior-season form
across the offseason would change the feature distribution and belongs in a
separately backtested Version 2.

Historical results live in `reports/`:

- `rolling_backtest.json`: expanding-window 2021–2025 evaluation
- `baseline_metrics.json`: locked Version 1 regression baseline
- `betting_retrospective.json`: threshold research
- `production_model.json`: public production-model manifest

Across the public 2022–2025 rolling test seasons, the no-wind model's combined
rule returned approximately 3.99% over 174 flat-stake signals. Both conditions were selected
retrospectively, so the signal is displayed as experimental rather than as a
promise of profitability.

## Local development

### Python API

```bash
python -m pip install -r requirements.txt
pytest -q
uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000` and its interactive OpenAPI
documentation at `http://localhost:8000/docs`.

### Next.js interface

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`.

### Full application with Docker

```bash
docker compose up --build
```

This starts the interface, API, and PostgreSQL. SQLite remains the automatic
fallback when `DATABASE_URL` is not set.

## Production workflow

For the complete Neon, Render, and Vercel setup, including the safe one-command
database seed, see [DEPLOYMENT.md](DEPLOYMENT.md).

Train and version the all-data model:

```bash
python -m scripts.train_production
```

Generate a weekly snapshot after both teams have five completed games:

```bash
python -m scripts.generate_week 2026 6
```

Load schedule-only cards for eligible future matchups without creating fake
predictions:

```bash
python -m scripts.load_upcoming_schedule
```

The generator fetches the hourly Open-Meteo forecast nearest kickoff for the
website, but weather is not an official model input. The API labels
retractable-roof status instead of claiming it is known. Neutral-site games
require explicit stadium coordinates so a home-team location is never silently
substituted.

Refresh the conditions displayed by the API without changing the locked
prediction:

```bash
python -m scripts.update_weather 2026 6
```

Import completed games and automatically refresh the live performance table:

```bash
python -m scripts.sync_results
```

The command derives the prospective season from the production model manifest,
so it does not require a code change during the season. It is idempotent and is
suitable for a scheduled job after games finish.

Backfill archived kickoff conditions for the 2022–2025 historical showcase:

```bash
python -m scripts.backfill_historical_weather
```

The backfill uses explicit coordinates for neutral-site and international
venues, caches the downloaded conditions, and can safely be rerun without
changing predictions or results.

This command is suitable for a scheduled job (for example, every 10 minutes on
game day). The database records the provider, forecast target, and retrieval
time. Open-Meteo requires no API key for this non-commercial project.

Build the historical showcase once. This caches a rolling model for every test
season and stores Weeks 6–18 for browsing without retraining during web
requests:

```bash
python -m scripts.build_historical_showcase
```

To attach market prices, normalize provider output to the contract in
`src/market.py` and pass it as a CSV:

```bash
python -m scripts.generate_week 2026 6 --market-csv data/market_week_06.csv
```

The odds key belongs only in the server-side `ODDS_API_KEY` environment
variable. It must never be committed or exposed to the browser. A live provider
account still needs to be connected before automatic market ingestion can run.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health |
| `GET /seasons` | Seasons with stored predictions |
| `GET /predictions/{season}/{week}` | Latest immutable weekly snapshot |
| `GET /schedule/{season}/{week}` | Eligible schedule-only preview cards |
| `GET /model` | Production version and frozen configuration |
| `GET /performance` | Frozen backtests plus automatically calculated live-season results |

## Quality controls

```bash
pytest -q
cd web && npm run lint && npm run build
```

GitHub Actions runs both Python and frontend checks on pushes and pull
requests. Generated databases, trained binaries, secrets, caches, and
prediction artifacts are excluded from Git.

## Repository layout

```text
api/          FastAPI, SQLAlchemy models, persistence, schemas
src/          processing, training, evaluation, pregame, prediction contracts
scripts/      production training, rolling backtest, weekly generation
reports/      committed model manifests and historical results
tests/        Python unit and integration tests
web/          Next.js TypeScript application
```
