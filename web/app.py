import json, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prglm.inference import load,generate

ROOT=Path(__file__).resolve().parent.parent
RUN=Path(os.environ.get('PRGLM_RUN',ROOT/'runs/smoke'))
app=FastAPI(title='PRG-LM playground')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])
cache={}


class Request(BaseModel):
    prompt: str = 'Once upon a time'
    temperature: float = Field(1,gt=0,le=5)
    seed: int = 42
    max_tokens: int = Field(12,ge=1,le=100)
    recurrence: bool = True
    fatigue: bool = True
    max_cycles: int = Field(4,ge=1,le=16)
    stochastic: bool = True
    forced_region: int | None = None
    recurrence_override: float | None = Field(None,ge=0,le=1)
    fatigue_strength: float = Field(.5,ge=0,le=10)
    recovery: float = Field(.8,ge=0,le=1)
    repeats: int = Field(1,ge=1,le=100)


@app.get('/api/benchmarks')
def benchmarks():
    file=ROOT/'dashboard/results.json' if (ROOT/'dashboard/results.json').exists() else RUN/'results.json'
    return json.loads(file.read_text()) if file.exists() else {'results':[]}


@app.get('/sample.json')
def sample():
    file=ROOT/'dashboard/sample.json'
    if not file.exists(): raise HTTPException(404,'Export dashboard first')
    return FileResponse(file)


@app.post('/api/generate')
def api_generate(req:Request):
    if not (RUN/'prg.pt').exists() or not (RUN/'transformer.pt').exists():
        raise HTTPException(503,'Run training first: python -m prglm.experiment')
    settings=req.model_dump()
    repeats=settings.pop('repeats'); prompt=settings.pop('prompt')
    results=[]
    for i in range(repeats):
        pair={}
        for name in ('transformer','prg'):
            if name not in cache: cache[name]=load(RUN,name)
            model,tok=cache[name]
            args={k:v for k,v in settings.items() if k in {'temperature','seed','max_tokens'}}
            args['seed']=req.seed+i
            if name=='prg':
                args.update({k:v for k,v in settings.items() if k not in {'temperature','seed','max_tokens'}})
            pair[name]=generate(model,tok,prompt,**args)
        results.append(pair)
    return {'runs':results}


app.mount('/',StaticFiles(directory=ROOT/'web/static',html=True),name='static')
