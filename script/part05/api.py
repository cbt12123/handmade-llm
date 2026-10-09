"""Optional local single-user interface. Start one uvicorn worker only."""
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from knowledge import Knowledge
from model import HTTPModel
from runner import Agent
from store import Store
from toolbox import Toolbox
from presentation import present

app = FastAPI(title='Course learning agent')
gate = threading.Lock()


class TaskRequest(BaseModel):
    request: str = Field(min_length=1,max_length=1200)
    learner: str = Field(default='demo',pattern=r'^[a-zA-Z0-9_-]{1,40}$')
    allow_record: bool = False


@app.get('/health')
def health():
    return {'status':'ok','scope':'local single-user teaching interface; not authenticated'}


@app.post('/tasks')
def tasks(body:TaskRequest):
    if not gate.acquire(blocking=False):
        raise HTTPException(429,'另一个任务正在执行')
    store, model = None, None
    try:
        store = Store()
        model = HTTPModel()
        agent = Agent(model,Toolbox(Knowledge(),store),store)
        state = agent.run(agent.create(body.request,body.learner,body.allow_record))
        return present(state)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422,str(exc)) from exc
    finally:
        if store:
            store.close()
        if model:
            model.session.close()
        gate.release()
