import httpx
from app.core.metrics import QDRANT_UP


class GrafanaService:

    def __init__(self, grafana_url: str, prometheus_url: str) -> None:
        self.grafana_url = grafana_url
        self.prometheus_url = prometheus_url

    # ── Prometheus health ─────────────────────────────────────────────────────

    async def check_prometheus(self) -> dict:
        """
        Pings Prometheus /-/healthy endpoint.
        Returns status and reachability as a dictionary.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.prometheus_url}/-/healthy",
                    timeout=5.0,
                )
            return {
                "reachable": True,
                "status_code": response.status_code,
                "healthy": response.status_code == 200,
            }
        except httpx.RequestError:
            return {
                "reachable": False,
                "status_code": None,
                "healthy": False,
            }

    # ── Grafana health ────────────────────────────────────────────────────────

    async def check_grafana(self) -> dict:
        """
        Pings Grafana /api/health endpoint.
        Returns status and reachability as a dictionary.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.grafana_url}/api/health",
                    timeout=5.0,
                )
            return {
                "reachable": True,
                "status_code": response.status_code,
                "healthy": response.status_code == 200,
            }
        except httpx.RequestError:
            return {
                "reachable": False,
                "status_code": None,
                "healthy": False,
            }

    # ── Qdrant health ─────────────────────────────────────────────────────────

    async def check_qdrant(self, qdrant_url: str) -> dict:
        """
        Pings Qdrant /healthz endpoint and updates the QDRANT_UP gauge.
        This is the only place in the project that sets QDRANT_UP.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{qdrant_url}/healthz",
                    timeout=5.0,
                )
            is_healthy = response.status_code == 200
            QDRANT_UP.set(1 if is_healthy else 0)
            return {
                "reachable": True,
                "status_code": response.status_code,
                "healthy": is_healthy,
            }
        except httpx.RequestError:
            QDRANT_UP.set(0)
            return {
                "reachable": False,
                "status_code": None,
                "healthy": False,
            }

    # ── Full stack health ─────────────────────────────────────────────────────

    async def check_all(self, qdrant_url: str) -> dict:
        """
        Runs all three health checks and returns a combined report.
        Called by /monitoring/stats to give a full stack picture.
        """
        prometheus = await self.check_prometheus()
        grafana = await self.check_grafana()
        qdrant = await self.check_qdrant(qdrant_url)

        return {
            "prometheus": prometheus,
            "grafana": grafana,
            "qdrant": qdrant,
            "all_healthy": all([
                prometheus["healthy"],
                grafana["healthy"],
                qdrant["healthy"],
            ]),
        }