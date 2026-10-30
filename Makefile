.DEFAULT_GOAL := help
COMPOSE := docker compose
APP := $(COMPOSE) exec app

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

build:           ## build the app image
	$(COMPOSE) build

up:              ## start Elasticsearch + app (search works without any LLM)
	$(COMPOSE) up -d --build

up-airflow:      ## start Airflow stack (webserver, scheduler, postgres, redis)
	$(COMPOSE) --profile airflow up -d --build

down:            ## stop all containers (including Airflow if started)
	$(COMPOSE) --profile airflow down

down-airflow:    ## stop Airflow stack only
	$(COMPOSE) stop airflow-webserver airflow-scheduler airflow-triggerer airflow_init airflow_postgres airflow_redis

logs:            ## follow logs
	$(COMPOSE) logs -f --tail=100

logs-airflow:    ## follow Airflow logs
	$(COMPOSE) logs -f --tail=100 airflow-webserver airflow-scheduler

status:          ## Elasticsearch health and index stats
	$(APP) vaultlens status

sync:            ## incremental index of all vaults
	$(APP) vaultlens sync

reindex:         ## full reindex (builds notes_vN+1, swaps alias)
	$(APP) vaultlens reindex

test:            ## unit tests (offline: no Elasticsearch needed)
	$(COMPOSE) run --rm --no-deps app pytest -q

lint:            ## ruff
	$(COMPOSE) run --rm --no-deps app ruff check src tests

shell:           ## shell in the app container
	$(APP) bash

clean:           ## DANGER: stop and delete all volumes (index, saved searches, db data)
	$(COMPOSE) --profile airflow down -v
