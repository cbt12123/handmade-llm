"""Small deterministic protocol exercise, explicitly NOT an LLM experiment."""
import json


def next_action(observation):
    if observation is None:
        return {'action':'multiply', 'args':{'a':3,'b':4}}
    return {'action':'finish', 'answer':f'计算结果是 {observation}'}


def main():
    observation = None
    tools = {'multiply':lambda a,b:a*b}
    for _ in range(3):
        action = next_action(observation)
        print(json.dumps(action,ensure_ascii=False))
        if action['action'] == 'finish':
            break
        observation = tools[action['action']](**action['args'])
    else:
        raise RuntimeError('未在预算内完成')


if __name__ == '__main__':
    main()
