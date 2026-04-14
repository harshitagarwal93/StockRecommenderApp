"""CosmosDB persistence layer with local JSON fallback for development."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

_LOCAL_DATA_DIR = Path(__file__).parent.parent / ".data"


class CosmosStore:
    """Thin wrapper around Azure Cosmos DB (or local JSON files for dev)."""

    def __init__(self, config: Config):
        self._config = config
        self._container = None
        self._use_local = False
        self._init_store()

    def _init_store(self) -> None:
        if not self._config.cosmos_endpoint:
            logger.info("No COSMOS_ENDPOINT — using local JSON file storage")
            self._use_local = True
            _LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            return

        try:
            if self._config.cosmos_key:
                from azure.cosmos import CosmosClient

                client = CosmosClient(
                    self._config.cosmos_endpoint, self._config.cosmos_key
                )
            else:
                from azure.cosmos import CosmosClient
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                client = CosmosClient(
                    self._config.cosmos_endpoint, credential=credential
                )

            db = client.create_database_if_not_exists(id=self._config.cosmos_database)
            self._container = db.create_container_if_not_exists(
                id=self._config.cosmos_container,
                partition_key={"paths": ["/type"], "kind": "Hash"},
                offer_throughput=400,
            )
            logger.info(
                "Connected to CosmosDB %s/%s",
                self._config.cosmos_database,
                self._config.cosmos_container,
            )
        except Exception:
            logger.exception("CosmosDB init failed — falling back to local storage")
            self._use_local = True
            _LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def upsert(self, item: dict) -> dict:
        if self._use_local:
            return self._local_upsert(item)
        try:
            return self._container.upsert_item(item)
        except Exception:
            logger.exception("CosmosDB upsert failed for %s", item.get("id"))
            return self._local_upsert(item)

    def read(self, item_id: str, partition_key: str) -> dict | None:
        if self._use_local:
            return self._local_read(item_id)
        try:
            return self._container.read_item(item=item_id, partition_key=partition_key)
        except Exception:
            return self._local_read(item_id)

    def query(self, query_text: str, parameters: list[dict] | None = None) -> list[dict]:
        if self._use_local:
            return self._local_query_all()
        try:
            items = self._container.query_items(
                query=query_text,
                parameters=parameters or [],
                enable_cross_partition_query=True,
            )
            return list(items)
        except Exception:
            logger.exception("CosmosDB query failed")
            return []

    def _local_path(self, item_id: str) -> Path:
        safe_name = item_id.replace("/", "_").replace("\\", "_")
        return _LOCAL_DATA_DIR / f"{safe_name}.json"

    def _local_upsert(self, item: dict) -> dict:
        path = self._local_path(item["id"])
        path.write_text(json.dumps(item, indent=2, default=str), encoding="utf-8")
        return item

    def _local_read(self, item_id: str) -> dict | None:
        path = self._local_path(item_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _local_query_all(self) -> list[dict]:
        results = []
        for p in _LOCAL_DATA_DIR.glob("*.json"):
            try:
                results.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return results
