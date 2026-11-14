"""Airflow DAG for VaultLens Full Zero-Downtime Reindex.

1. check_create_catalog_table: Checks/creates the PostgreSQL vault_notes table.
2. full_reindex_notes: Builds notes_vN+1 from all registered vaults, swaps the Elasticsearch
   alias atomically, and synchronizes note catalog metadata in PostgreSQL.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from vaultlens_pipeline.postgres import check_or_create_table
from vaultlens_pipeline.sync import reindex_vault_notes

default_args = {
    "owner": "VaultLens",
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="vaultlens_full_reindex",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    description=(
        "Full zero-downtime reindex of all markdown vaults into Elasticsearch "
        "(notes_vN+1 alias swap) and PostgreSQL catalog"
    ),
    tags=["vaultlens", "search", "elasticsearch", "postgres", "reindex"],
)

task_check_create_table = PythonOperator(
    task_id="check_create_catalog_table",
    python_callable=check_or_create_table,
    dag=dag,
)

task_reindex_notes = PythonOperator(
    task_id="full_reindex_notes",
    python_callable=reindex_vault_notes,
    dag=dag,
)

task_check_create_table >> task_reindex_notes
