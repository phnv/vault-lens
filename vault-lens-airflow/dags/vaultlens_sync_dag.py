"""Airflow DAG for VaultLens Note Ingestion and Search Synchronization.

1. check_create_catalog_table: Checks/creates the PostgreSQL vault_notes table.
2. sync_notes_and_index: Scans Markdown notes, incrementally updates Elasticsearch (notes alias),
   and synchronizes note catalog metadata in PostgreSQL.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from vaultlens_pipeline.postgres import check_or_create_table
from vaultlens_pipeline.sync import sync_vault_notes

default_args = {
    "owner": "VaultLens",
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="vaultlens_sync_notes",
    default_args=default_args,
    schedule_interval="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    description="Incrementally synchronizes personal markdown notes into Elasticsearch and PostgreSQL catalog (every 15 minutes)",
    tags=["vaultlens", "search", "elasticsearch", "postgres", "incremental"],
)

task_check_create_table = PythonOperator(
    task_id="check_create_catalog_table",
    python_callable=check_or_create_table,
    dag=dag,
)

task_sync_notes = PythonOperator(
    task_id="sync_notes_and_index",
    python_callable=sync_vault_notes,
    dag=dag,
)

task_check_create_table >> task_sync_notes
