"""SQLite task checkpoints and learning records; transactions live in the runner."""
import json
import re
import sqlite3
import os
from pathlib import Path
from common import OUT


class Store:
    def __init__(self, path=None):
        path = Path(path or os.environ.get('AGENT_DB_PATH', OUT / 'learning.sqlite3'))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, learner TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                task TEXT NOT NULL REFERENCES tasks(id), step INTEGER NOT NULL,
                learner TEXT NOT NULL, question TEXT NOT NULL,
                correct INTEGER NOT NULL, feedback TEXT NOT NULL,
                PRIMARY KEY(task, step));
        ''')

    @staticmethod
    def learner_id(value):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', value):
            raise ValueError('learner 必须是 1～40 位英文字母、数字、下划线或短横线')
        return value

    def checkpoint(self, state):
        self.db.execute('INSERT INTO tasks VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state',
                        (state['task'], state['learner'], json.dumps(state, ensure_ascii=False)))

    def load(self, task, learner):
        row = self.db.execute('SELECT learner,state FROM tasks WHERE id=?', (task,)).fetchone()
        if row is None or row[0] != learner:
            raise ValueError('任务不存在或不属于当前学习者')
        return json.loads(row[1])

    def recent(self, learner):
        rows = self.db.execute('SELECT question,correct,feedback FROM records WHERE learner=? ORDER BY rowid DESC LIMIT 5',
                               (learner,)).fetchall()
        return [{'question_id':q, 'correct':bool(c), 'feedback':f} for q,c,f in rows]

    def record(self, state, grade):
        self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?)',
                        (state['task'], state['step'], state['learner'], grade['question_id'],
                         int(grade['correct']), grade['feedback']))
        return {'saved':True, 'question_id':grade['question_id'], 'correct':grade['correct']}

    def close(self):
        self.db.close()
