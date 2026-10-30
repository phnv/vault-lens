FROM apache/airflow:2.10.4

USER airflow
ENV AIRFLOW_HOME=/opt/airflow/vault-lens-airflow

USER root
RUN mkdir -p /opt/airflow/vault-lens-airflow \
    && chown -R airflow:0 /opt/airflow/vault-lens-airflow \
    && chmod -R g+rwX /opt/airflow/vault-lens-airflow
USER airflow

WORKDIR /opt/airflow/vault-lens-airflow
COPY requirements.txt /opt/airflow/vault-lens-airflow/requirements.txt
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000,gid=0 \
    pip install --default-timeout=120 --retries 5 apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
