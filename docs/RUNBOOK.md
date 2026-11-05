# House Price API: Runbook

## Service level objectives (measured over 7 days)

| SLO | Target | Where to see it |
|---|---|---|
| Availability | ≥ 99% of /predict requests return non-5xx | Grafana: requests by status |
| Latency | p95 /predict < 200 ms | Grafana: p95 latency; HighLatency alert |
| Monitoring freshness | monitoring_job succeeded in the last 26 h | Dagster: runs |
| Model quality | live RMSE ≤ 1.5 × holdout RMSE | Grafana: latest monitoring checks |

## Daily operations

- Start: `docker compose up -d`. Stop: `docker compose down` (never add `-v` unless you mean to delete all data)
- Status: `docker compose ps`, `docker compose exec api mlops-model status`
- Logs: `docker compose logs -f api` (JSON; search by `request_id`)

## Release a new model

1. Run `training_job` in Dagster (or wait for Monday's scheduled run).
2. In MLflow, read the new version's `model_card.md` and gate tags.
3. `docker compose exec api mlops-model promote`
4. `docker compose restart api`
5. `uv run python scripts/smoke_test.py --api-key <key>`: every line must be PASS.
6. Watch the dashboard for 15 minutes.

## Roll back a model

1. `docker compose exec api mlops-model rollback` (or `--version <n>`)
2. `docker compose restart api`
3. Smoke test. Confirm `/ready` shows the expected version.

## Alert → action

| Alert / signal | First checks | Typical fix |
|---|---|---|
| ApiDown | `docker compose ps`, `docker compose logs api` | Restart api; check memory limits |
| ModelNotLoaded | `docker compose logs api`; is MLflow healthy? does a champion exist? | Fix MLflow, promote a version, restart api |
| HighErrorRate | API logs for exceptions (search `"level": "ERROR"`) | Roll back the model if it started after a release |
| HighLatency | CPU/memory (`docker stats`); is a training run active? | Wait for training to finish; investigate slow requests |
| PredictionLogFailures | `docker compose ps postgres`, Postgres logs | Restart Postgres; check disk space |
| input_drift failed | Open the HTML drift report in `artifacts/reports/` | Investigate which features drifted; retrain if the change is real |
| live_rmse failed | Compare predictions and feedback in Postgres | Retrain on recent data; roll back if a recent release caused it |
| Validation failed in training_job | `artifacts/reports/validation.json` | Fix or reject the source data. Never loosen a rule just to make it pass |

## Backups

    docker compose exec -T postgres pg_dumpall -U postgres > backup-$(date +%F).sql
    docker run --rm -v modern-mlops_mlflow-artifacts:/data -v "$PWD":/backup alpine:3.20 \
      tar czf /backup/mlflow-artifacts-$(date +%F).tgz -C /data .

Restore by recreating the volumes and loading the dump. Practise a restore at least once.

## Rotate an API key

1. Add the new key to `API_KEYS` (comma-separated), `docker compose up -d api`.
2. Move clients to the new key.
3. Remove the old key, `docker compose up -d api`.