# Portfolio deployment

This repository is prepared for a three-service portfolio deployment:

- Vercel serves the Next.js interface from `web/`.
- Render serves the read-only FastAPI application from the root `Dockerfile`.
- Neon stores predictions, results, conditions, and schedule cards in Postgres.

The public API does not train models. Offline jobs publish immutable records to
Postgres, which keeps web requests lightweight and prevents a free host restart
from deleting data.

## 1. Create the Neon database

Create a Neon project and copy its pooled connection string. SQLAlchemy should
use the psycopg driver, so change the URL prefix from `postgresql://` to
`postgresql+psycopg://` if necessary.

Keep this value private. Locally, place it in an untracked `.env` file as:

```dotenv
DEPLOY_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

Seed Neon from the verified local portfolio database:

```bash
set -a
source .env
set +a
python -m scripts.seed_deployment_database
```

The seed command is idempotent. It copies only final no-wind historical
snapshots plus prospective-season records, results, weather, and schedule
cards. It never deletes deployment data.

## 2. Deploy the API on Render

In Render, create a Blueprint from the GitHub repository. Render reads the
root `render.yaml` and builds the root `Dockerfile`.

Provide these secret values when prompted:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
CORS_ORIGINS=https://YOUR-PROJECT.vercel.app
```

After deployment, verify:

```text
https://fourth-down-forecast-api.onrender.com/health
https://fourth-down-forecast-api.onrender.com/docs
```

The health check verifies both the API process and its database connection.
Render's free web service can sleep after inactivity, so its first request can
be slower. Upgrading only the API service later removes that portfolio-demo
tradeoff without changing the architecture.

## 3. Deploy the interface on Vercel

Import the same GitHub repository into Vercel and configure:

- Root Directory: `web`
- Framework Preset: Next.js
- Environment variable `API_URL`: the public Render URL, without a trailing
  slash

Deploy, then update `CORS_ORIGINS` in Render with the final Vercel domain if it
differs from the value entered earlier.

The public link will look like:

```text
https://fourth-down-forecast.vercel.app
```

Vercel redeploys the interface after pushes to the connected production
branch. Render does the same for the API.

## 4. Production data updates

The persistent target for every update command is the same Neon connection
string:

```bash
python -m scripts.load_upcoming_schedule --database-url "$DEPLOY_DATABASE_URL"
python -m scripts.generate_week 2026 6 --database-url "$DEPLOY_DATABASE_URL"
python -m scripts.update_weather 2026 6 --database-url "$DEPLOY_DATABASE_URL"
python -m scripts.sync_results --database-url "$DEPLOY_DATABASE_URL"
```

Prediction persistence enforces the per-game one-hour lock. Weather and final
results remain independently updateable after a prediction is frozen.

Live odds ingestion and scheduled execution require `ODDS_API_KEY` and the
automation workflow described in the production README. Until those are
connected, weekly generation remains an explicit operator command.

## Deployment checks

- `/health` returns HTTP 200.
- `/performance` ends with the prospective 2026 row.
- Historical API pages total 761 final tie-adjusted no-wind predictions and 53 signals.
- The Vercel Methodology API link opens the Render OpenAPI documentation.
- No database URLs or API keys appear in Git, browser JavaScript, or build logs.
