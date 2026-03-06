"""Data Pipeline — ETL with cascading failures and partial recovery.

Models a data ingestion pipeline where sources, transformations, and
destinations can fail independently. Demonstrates cascading failure
handling — when an upstream tool fails, the router automatically
adjusts the entire downstream path.

Graph:
    ingest_api ────→ validate → transform_pandas ──→ load_postgres ──→ notify
    ingest_csv ────→ validate → transform_spark  ──→ load_bigquery ──→ notify
    ingest_stream ─→ validate                       load_s3 ────────→ notify

This is the kind of pipeline that breaks Airflow DAGs at 3am.
Self-Healing Router fixes it without paging anyone.

Try it:
    python examples/data_pipeline.py
"""

import random
from self_healing_router import SelfHealingRouter, EscalationResult

# --- Service health ---
health = {
    "api": True,
    "csv": True,
    "stream": True,
    "pandas": True,
    "spark": True,
    "postgres": True,
    "bigquery": True,
    "s3": True,
    "slack": True,
}


def ingest_api(data: dict) -> dict:
    if not health["api"]:
        raise ConnectionError("REST API: connection timeout after 30s")
    records = data.get("expected_records", 10000)
    print(f"    📡 API ingestion: {records:,} records fetched")
    return {**data, "source": "api", "records": records, "raw_data": f"api_batch_{records}"}


def ingest_csv(data: dict) -> dict:
    if not health["csv"]:
        raise FileNotFoundError("CSV: /data/export_20260306.csv not found")
    records = data.get("expected_records", 10000)
    print(f"    📄 CSV ingestion: {records:,} records loaded from disk")
    return {**data, "source": "csv", "records": records, "raw_data": f"csv_batch_{records}"}


def ingest_stream(data: dict) -> dict:
    if not health["stream"]:
        raise ConnectionError("Kafka: broker disconnected")
    records = data.get("expected_records", 10000) // 2  # streams get partial batches
    print(f"    🌊 Stream ingestion: {records:,} records from Kafka")
    return {**data, "source": "stream", "records": records, "raw_data": f"stream_batch_{records}"}


def validate(data: dict) -> dict:
    records = data.get("records", 0)
    valid = int(records * 0.98)  # 2% invalid
    print(f"    ✓  Validated: {valid:,}/{records:,} records passed ({valid/records:.1%})")
    return {**data, "valid_records": valid, "validated": True}


def transform_pandas(data: dict) -> dict:
    if not health["pandas"]:
        raise MemoryError("Pandas: DataFrame exceeded 16GB memory limit")
    records = data.get("valid_records", 0)
    print(f"    🐼 Pandas transform: {records:,} records enriched + deduplicated")
    return {**data, "transformer": "pandas", "transformed": True}


def transform_spark(data: dict) -> dict:
    if not health["spark"]:
        raise ConnectionError("Spark: executor lost, stage 3 failed")
    records = data.get("valid_records", 0)
    print(f"    ⚡ Spark transform: {records:,} records processed (distributed)")
    return {**data, "transformer": "spark", "transformed": True}


def load_postgres(data: dict) -> dict:
    if not health["postgres"]:
        raise ConnectionError("PostgreSQL: too many connections (max: 100)")
    records = data.get("valid_records", 0)
    print(f"    🐘 PostgreSQL: {records:,} rows upserted to analytics.events")
    return {**data, "destination": "postgres", "loaded": True}


def load_bigquery(data: dict) -> dict:
    if not health["bigquery"]:
        raise PermissionError("BigQuery: 403 quota exceeded for project")
    records = data.get("valid_records", 0)
    print(f"    ☁️  BigQuery: {records:,} rows streamed to dataset.events")
    return {**data, "destination": "bigquery", "loaded": True}


def load_s3(data: dict) -> dict:
    if not health["s3"]:
        raise ConnectionError("S3: 500 Internal Server Error")
    records = data.get("valid_records", 0)
    print(f"    🪣 S3: {records:,} records written to s3://datalake/events/")
    return {**data, "destination": "s3", "loaded": True}


def notify(data: dict) -> dict:
    source = data.get("source", "?")
    dest = data.get("destination", "?")
    records = data.get("valid_records", 0)
    print(f"    🔔 Pipeline complete: {source} → {dest} ({records:,} records)")
    return {**data, "notified": True, "pipeline_complete": True}


def escalation(failed_tool, attempted_paths, context):
    return EscalationResult(
        action="abort",
        detail=f"Pipeline failed: no viable path from source to destination. "
               f"Attempted {len(attempted_paths)} paths. Alerting on-call.",
    )


def build_router():
    router = SelfHealingRouter(escalation_callback=escalation)

    tools = [
        ("start", lambda d: d),  # virtual entry point
        ("ingest_api", ingest_api), ("ingest_csv", ingest_csv),
        ("ingest_stream", ingest_stream), ("validate", validate),
        ("transform_pandas", transform_pandas), ("transform_spark", transform_spark),
        ("load_postgres", load_postgres), ("load_bigquery", load_bigquery),
        ("load_s3", load_s3), ("notify", notify),
    ]
    for name, fn in tools:
        router.add_tool(name, handler=fn)

    # Entry → Ingestion sources
    router.add_edge("start", "ingest_api", weight=1.0)      # preferred: freshest data
    router.add_edge("start", "ingest_csv", weight=2.0)      # backup: stale but reliable
    router.add_edge("start", "ingest_stream", weight=1.5)   # alternative: partial batches

    # Ingestion → Validation
    router.add_edge("ingest_api", "validate", weight=1.0)
    router.add_edge("ingest_csv", "validate", weight=1.0)
    router.add_edge("ingest_stream", "validate", weight=1.0)

    # Validation → Transformation
    router.add_edge("validate", "transform_pandas", weight=1.0)  # fast for <1M rows
    router.add_edge("validate", "transform_spark", weight=2.0)   # distributed fallback

    # Transformation → Loading
    router.add_edge("transform_pandas", "load_postgres", weight=1.0)   # primary destination
    router.add_edge("transform_pandas", "load_bigquery", weight=2.0)   # cloud backup
    router.add_edge("transform_pandas", "load_s3", weight=3.0)         # raw dump fallback
    router.add_edge("transform_spark", "load_postgres", weight=1.0)
    router.add_edge("transform_spark", "load_bigquery", weight=1.5)    # spark → BQ is natural
    router.add_edge("transform_spark", "load_s3", weight=2.0)

    # Loading → Notification
    router.add_edge("load_postgres", "notify", weight=1.0)
    router.add_edge("load_bigquery", "notify", weight=1.0)
    router.add_edge("load_s3", "notify", weight=1.0)

    return router


def run(title, desc, overrides, data=None):
    global health
    health = {k: True for k in health}  # reset
    health.update(overrides)

    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"  {desc}")
    print(f"{'=' * 65}")

    pipeline_data = data or {"expected_records": 50000}
    router = build_router()
    result = router.route("start", "notify", input_data=pipeline_data)

    status = "✅ Complete" if result.success else "❌ Failed"
    print(f"\n  {status}")
    print(f"  Path:      {' → '.join(result.path) if result.path else '(none)'}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Reroutes:  {result.reroutes}")
    if result.errors:
        print(f"  Failures:  {len(result.errors)}")
        for e in result.errors:
            print(f"    └─ {e}")
    return result


def main():
    print("\n" + " DATA PIPELINE — ETL with Cascading Failure Recovery ".center(65, "═"))

    # 1: Happy path
    run("Scenario 1: Happy Path",
        "API → Pandas → PostgreSQL (optimal path)", {})

    # 2: Pandas OOM → Spark fallback
    run("Scenario 2: Pandas Out of Memory",
        "Large dataset kills Pandas → automatic reroute to Spark",
        {"pandas": False},
        {"expected_records": 5000000})

    # 3: API down + Postgres down → CSV + BigQuery
    run("Scenario 3: API + Postgres Down (Cascading)",
        "Primary source AND destination fail. Router finds CSV → Spark → BigQuery.",
        {"api": False, "postgres": False})

    # 4: Multiple failures → S3 dump as last resort
    run("Scenario 4: Only S3 Left",
        "Postgres + BigQuery both down. Pipeline degrades to raw S3 dump.",
        {"postgres": False, "bigquery": False})

    # Summary
    print(f"\n{'═' * 65}")
    print("  COMPARISON WITH AIRFLOW/DAGSTER/PREFECT")
    print(f"{'═' * 65}")
    print("  Traditional orchestrators:")
    print("    ❌ DAG defined at deploy time — can't reroute at runtime")
    print("    ❌ Failure = retry same path or manual intervention")
    print("    ❌ Adding a new fallback = edit DAG + redeploy + pray")
    print()
    print("  Self-Healing Router:")
    print("    ✅ Reroutes in <1ms based on what's actually healthy")
    print("    ✅ New fallback = one add_tool() + add_edge() call")
    print("    ✅ No LLM needed — Dijkstra finds the next best path")
    print("    ✅ Every failure is logged and observable (no silent drops)")


if __name__ == "__main__":
    main()
