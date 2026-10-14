"""HTTP attempt accounting without authentication headers or prompt text."""
import json
import time
import uuid
from .audit import write_audit


def make_http_cost_hooks(directory, judge_tag):
    async def on_request(request):
        stamp={"attempt_id":uuid.uuid4().hex,"started_at":time.time(),"started_monotonic":time.monotonic()}
        request.extensions["rar_cost_audit"]=stamp
        try: model=json.loads(request.content).get("model")
        except (ValueError,AttributeError): model=None
        write_audit(directory,{"event":"request","judge":judge_tag,"attempt_id":stamp["attempt_id"],
            "started_at":stamp["started_at"],"host":request.url.host,"path":request.url.path,
            "model":model,"sdk_retry_count":request.headers.get("x-stainless-retry-count")})

    async def on_response(response):
        stamp=response.request.extensions.get("rar_cost_audit",{})
        record={"event":"response","judge":judge_tag,"attempt_id":stamp.get("attempt_id"),
                "started_at":stamp.get("started_at"),"status_code":response.status_code,
                "request_id":response.headers.get("x-request-id"),
                "elapsed_seconds":time.monotonic()-stamp.get("started_monotonic",time.monotonic())}
        try:
            await response.aread()
            payload=response.json()
            if isinstance(payload,dict):
                record.update(response_id=payload.get("id"),model=payload.get("model"),provider=payload.get("provider"),usage=payload.get("usage"))
                error=payload.get("error")
                if isinstance(error,dict):record["error"]={k:error.get(k) for k in ("code","type")}
        except Exception as exc:
            record["response_read_error"]=type(exc).__name__
            write_audit(directory,record)
            raise
        write_audit(directory,record)
    return {"request":[on_request],"response":[on_response]}
