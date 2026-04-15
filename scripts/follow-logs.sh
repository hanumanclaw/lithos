#!/usr/bin/env bash
# Follow docker compose logs with health check noise filtered out.
# Run from the directory containing docker-compose.yml (e.g. docker/).

if [ ! -f docker-compose.yml ] && [ ! -f docker-compose.yaml ] && [ ! -f compose.yml ] && [ ! -f compose.yaml ]; then
  echo "Error: No docker compose file found in the current directory." >&2
  echo "Run this script from the directory containing docker-compose.yml (e.g. cd docker/)" >&2
  exit 1
fi

docker compose logs -f 2>&1 | grep -v -E 'GET /health|Batches:.*it/s'
