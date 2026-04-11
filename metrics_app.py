import os
import httpx
import ray
from fastapi import FastAPI
from fastapi.responses import Response
from ray import serve

RAY_METRICS_PORT = int(os.environ.get("RAY_METRICS_PORT", "8080"))

fastapi_app = FastAPI()

@serve.deployment(name="metrics-ingress", num_replicas=1)
@serve.ingress(fastapi_app)
class MetricsIngress:
    def __init__(self):
        self._http = httpx.AsyncClient(timeout=10.0)

        # Resolve head node IP via GCS address (GCS always runs on the head node)
        gcs_address = ray.get_runtime_context().gcs_address  # "HEAD_IP:PORT"
        head_ip = gcs_address.split(":")[0]
        self._metrics_url = f"http://{head_ip}:{RAY_METRICS_PORT}/metrics"

    @fastapi_app.get("/metrics")
    async def metrics(self):
        try:
            resp = await self._http.get(self._metrics_url)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except httpx.RequestError as exc:
            return Response(
                content=f"# ERROR: could not reach internal metrics: {exc}\n",
                status_code=200,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

app = MetricsIngress.bind()