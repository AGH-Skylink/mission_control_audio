
from fastapi import FastAPI
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

    @app.post("/config")
    def update_config(cfg: ConfigModel):
        engine.reload_config(cfg.model_dump())
        return {"ok": True}

    @app.post("/ptt")
    def ptt(ptt: PttModel):
        engine.set_ptt(ptt.channel, mute=ptt.mute, gate_open=ptt.gate_open)
        return {"ok": True}

    return app


def run_api(app, host: str, port: int):
    uvicorn.run(app, host=host, port=port, log_level="info")
