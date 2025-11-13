from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
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

        @field_validator("sample_rate")
        @classmethod
        def _sr(cls, v):
            if v <= 0:
                raise ValueError("sample_rate must be > 0")
            return v

        @field_validator("channels")
        @classmethod
        def _ch(cls, v):
            if v not in (1, 2):
                raise ValueError("channels must be 1 or 2")
            return v

        @field_validator("sample_format")
        @classmethod
        def _fmt(cls, v):
            if v != "int16":
                raise ValueError("sample_format must be 'int16'")
            return v

        @field_validator("blocksize")
        @classmethod
        def _bs(cls, v):
            if v <= 0:
                raise ValueError("blocksize must be > 0")
            return v

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
            # DSP spec not satisfied → 500 with details
            raise HTTPException(status_code=500, detail=result)
        return result

    @app.post("/config")
    def update_config(cfg: ConfigModel):
        cfgdict = cfg.model_dump()
        # 1) validate inputs (no side effects)
        try:
            engine.validate_config(cfgdict)
        except ValueError as ve:
            # bad input → 400
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # unexpected validator error → 500
            raise HTTPException(status_code=500, detail=f"validation error: {e}")
        # 2) apply config
        try:
            engine.reload_config(cfgdict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to apply config: {e}")
        return {"status": "ok"}

    @app.post("/ptt")
    def ptt(ptt: PttModel):
        # check channel exists
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
