"""Read-only live log stream for the private full-scale notebook."""
import json
from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

request = ApiGetKernelSessionLogsStreamRequest()
request.user_name = "jonsmirl"
request.kernel_slug = "ssa-tail-fullscale-rtx6000"
request.wait_for_logs_url_seconds = 30
with api.build_kaggle_client() as client:
    response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
    if "text/event-stream" in response.headers.get("Content-Type", ""):
        for line in response.iter_lines():
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("data:"):
                value = line[5:].strip()
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
                print(value.get("data", value) if isinstance(value, dict) else value, flush=True)
    else:
        try:
            entries = response.json()
            for entry in entries[-20:]:
                print(entry.get("data", entry) if isinstance(entry, dict) else entry, flush=True)
        except (ValueError, TypeError):
            print(response.text[-12000:], flush=True)
