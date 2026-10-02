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
V1_RUN=Path(os.environ.get('PRGLM_V1_RUN',ROOT/'runs/v1_final/seed42'))
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
    accumulation: bool = True
    comparison: str = Field('legacy',pattern='^(legacy|byte-v1)$')


@app.get('/api/benchmarks')
def benchmarks():
    file=ROOT/'dashboard/results.json' if (ROOT/'dashboard/results.json').exists() else RUN/'results.json'
    return json.loads(file.read_text()) if file.exists() else {'results':[]}


@app.get('/sample.json')
def sample():
    file=ROOT/'dashboard/sample.json'
    if not file.exists(): raise HTTPException(404,'Export dashboard first')
    return FileResponse(file)


@app.get('/api/benchmarks_v1')
def benchmarks_v1():
    file=ROOT/'dashboard/v1_results.json'
    if not file.exists():raise HTTPException(404,'Run PRG-v1 benchmark and export dashboard')
    return json.loads(file.read_text())


@app.get('/v1_sample.json')
def sample_v1():
    file=ROOT/'dashboard/v1_sample.json'
    if not file.exists():raise HTTPException(404,'Export PRG-v1 dashboard')
    return FileResponse(file)


@app.get('/api/learning_curve')
def learning_curve():
    file=ROOT/'research/phase_a/learning_curve.json'
    if not file.exists():raise HTTPException(404,'Export Phase A results first')
    return FileResponse(file)


@app.get('/api/benchmarks_v2')
def benchmarks_v2():
    file=ROOT/'research/v2/results.json'
    if not file.exists():raise HTTPException(404,'Export v2 results first')
    return FileResponse(file)


class V2Request(Request):
    forced_region: int | None = Field(None,ge=0,le=31)
    checkpoint_seed: int = Field(42,ge=42,le=44)
    training_tokens: int = Field(819200,ge=12800,le=819200)
    topology_model: str = Field('prg_v2_adaptive',pattern='^prg_v2_(adaptive|uniform|no_recurrence|no_accumulation)$')


@app.post('/api/v2/generate')
def generate_v2(req:V2Request):
    from prglm.v2_inference import load_checkpoint
    roots={'transformer_core':ROOT/'runs/phase_a','transformer_total':ROOT/'runs/phase_a','prg_v1':ROOT/'runs/phase_a',
           req.topology_model:ROOT/'runs/mutable_v2'}
    paths={name:root/f'seed{req.checkpoint_seed}'/name/f'model_tokens_{req.training_tokens}.pt'
           for name,root in roots.items()}
    if req.checkpoint_seed==42 and req.training_tokens==819200:
        for name,path in list(paths.items()):
            bundled=ROOT/'checkpoints/byte256_seed42_819200'/f'{name}.pt'
            if bundled.exists():paths[name]=bundled
    if not all(p.exists() for p in paths.values()):raise HTTPException(503,'Selected checkpoint is unavailable. The bundled seed42 final models work without training; other choices need local runs.')
    result=[]
    for repeat in range(req.repeats):
        outputs={}
        for name,path in paths.items():
            key=('v2',str(path))
            if key not in cache:cache[key]=load_checkpoint(path)
            model,tokenizer=cache[key]
            settings={k:v for k,v in req.model_dump().items() if k in
                {'recurrence','accumulation','fatigue','max_cycles','forced_region','stochastic',
                 'recurrence_override','fatigue_strength','recovery'}} if name.startswith('prg') else {}
            outputs[name]=generate(model,tokenizer,req.prompt,max_tokens=req.max_tokens,
                temperature=req.temperature,seed=req.seed+repeat,**settings)
            if name.startswith('prg_v2'):
                outputs[name]['gateway_table']=model.gateway_table.cpu().tolist()
        result.append(outputs)
    return {'runs':result}


@app.post('/api/generate')
def api_generate(req:Request):
    root=V1_RUN if req.comparison=='byte-v1' else RUN
    names=('transformer_core','prg','prg_v1') if req.comparison=='byte-v1' else ('transformer','prg')
    if not all((root/f'{name}.pt').exists() for name in names):
        raise HTTPException(503,'Train the selected comparison and provide its run directory')
    settings=req.model_dump()
    repeats=settings.pop('repeats'); prompt=settings.pop('prompt');settings.pop('comparison')
    results=[]
    for i in range(repeats):
        pair={}
        for name in names:
            key=(str(root),name)
            if key not in cache: cache[key]=load(root,name)
            model,tok=cache[key]
            args={k:v for k,v in settings.items() if k in {'temperature','seed','max_tokens'}}
            args['seed']=req.seed+i
            if name=='prg':
                args.update({k:v for k,v in settings.items() if k not in {'temperature','seed','max_tokens','accumulation'}})
            if name=='prg_v1':
                args.update({k:v for k,v in settings.items() if k in {'recurrence','fatigue','accumulation','max_cycles','forced_region','stochastic','recurrence_override','fatigue_strength','recovery'}})
            pair['transformer' if name=='transformer_core' else name]=generate(model,tok,prompt,**args)
        results.append(pair)
    return {'runs':results}


app.mount('/',StaticFiles(directory=ROOT/'web/static',html=True),name='static')
