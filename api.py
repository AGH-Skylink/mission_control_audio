from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


def create_app(engine):
    app = FastAPI(title="Mission Control Audio Engine API")

    class ConfigModel(BaseModel):
        sample_rate: int
        sample_format: str
        channels: int
        blocksize: int
        logical_channels: dict
        dsp: dict

    class PttModel(BaseModel):
        channel: int
        mute: bool = False
        gate_open: bool = True

    @app.get("/status")
    def status():
        return engine.get_status()

    @app.get("/vu")
    def vu():
        return engine.get_vu_levels()

    @app.get("/self-check")
    def self_check():
        try:
            result = engine.self_check_dsp()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"self-check failed: {e}")
        if not (result.get("compressor_ok") and result.get("limiter_ok")):
            raise HTTPException(status_code=500, detail=result)
        return result

    @app.post("/config")
    def update_config(cfg: ConfigModel):
        # Pydantic v1: .dict(), v2: .model_dump()
        try:
            cfgdict = cfg.dict()
        except AttributeError:
            cfgdict = cfg.model_dump()

        # 1) validate first (no side effects)
        try:
            engine.validate_config(cfgdict)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"validation error: {e}")

        # 2) apply config
        try:
            engine.reload_config(cfgdict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to apply config: {e}")

        return {"status": "ok"}

    @app.post("/ptt")
    def ptt(ptt: PttModel):
        if str(ptt.channel) not in engine.channel_keys():
            raise HTTPException(status_code=400, detail=f"unknown channel {ptt.channel}")
        try:
            engine.set_ptt(ptt.channel, mute=ptt.mute, gate_open=ptt.gate_open)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to set ptt: {e}")
        return {"status": "ok"}

    return app


def run_api(app, host: str, port: int):
    uvicorn.run(app, host=host, port=port, log_level="info")
