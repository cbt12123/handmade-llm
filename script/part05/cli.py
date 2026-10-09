"""CLI front end; the caller, never the model, grants recording permission."""
import argparse
import json
from common import save
from knowledge import Knowledge
from model import HTTPModel
from runner import Agent
from store import Store
from toolbox import Toolbox
from presentation import present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', nargs='?')
    parser.add_argument('--learner', default='demo')
    parser.add_argument('--allow-record', action='store_true')
    parser.add_argument('--require-tool',action='append',default=[],choices=['search','quiz','grade','record','progress'])
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--url', default='http://127.0.0.1:8002')
    parser.add_argument('--model', default='tutorial-agent')
    parser.add_argument('--backend', choices=['vllm','tutorial'], default='vllm')
    args = parser.parse_args()
    if bool(args.resume) == bool(args.request):
        parser.error('提供问题或 --resume 任务ID，二者选一')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after 至少为 1')
    store = Store()
    try:
        model = HTTPModel(args.url,args.model,args.backend)
        agent = Agent(model,Toolbox(Knowledge(),store),store)
        state = store.load(args.resume,args.learner) if args.resume else agent.create(args.request,args.learner,args.allow_record,args.require_tool)
        state = agent.run(state,args.stop_after)
        save('last_task.json', {'state':state,'model_calls':model.calls})
        print(json.dumps(present(state),ensure_ascii=False,indent=2))
        if state['status'] in ('failed','exhausted'):
            raise SystemExit(1)
    finally:
        store.close()
        if 'model' in locals():
            model.session.close()


if __name__ == '__main__':
    main()
