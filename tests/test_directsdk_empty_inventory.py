"""Tool-free calls skip the inventory child while preserving native isolation."""
import json
import os
from pathlib import Path
import sys

import pytest
import directsdk


FAKE = '''import json, pathlib, sys
args = sys.argv
config = json.loads(args[args.index('--mcp-config') + 1])
assert '--strict-mcp-config' in args
assert args[args.index('--tools') + 1] == ''
assert args[args.index('--setting-sources') + 1] == ''
assert args[args.index('--permission-mode') + 1] == 'dontAsk'
assert args[args.index('--max-turns') + 1] == '1'
assert '--no-session-persistence' in args
inventory = config['mcpServers'].get('hermes')
manifest = json.loads(pathlib.Path(inventory['args'][-1]).read_text()) if inventory else None
pathlib.Path(sys.argv[1]).write_text(json.dumps({'servers': list(config['mcpServers']), 'manifest': manifest}))
for line in sys.stdin: pass
print(json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}], 'id': 'm', 'stop_reason': 'end_turn'}}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'subtype': 'success', 'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
'''


@pytest.mark.parametrize('tools', [None, [], [{'type': 'function', 'function': {
    'name': 'probe', 'description': 'Inventory only', 'parameters': {'type': 'object'},
}}]])
def test_inventory_child_is_needed_only_when_tools_exist(tmp_path, tools):
    script = tmp_path / 'native.py'
    script.write_text(FAKE)
    observed = tmp_path / 'observed.json'
    client = directsdk.Client(command=[sys.executable, str(script), str(observed)],
                              env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path)})
    try:
        chunks = list(client.create(model='sonnet', tools=tools, stream=True,
                                   messages=[{'role': 'user', 'content': 'go'}], timeout=5))
        assert chunks[-1].choices[0].finish_reason == 'stop'
        result = json.loads(observed.read_text())
        if tools:
            assert result['servers'] == ['hermes']
            assert result['manifest'][0]['name'] == 'probe'
        else:
            assert result == {'servers': [], 'manifest': None}
    finally:
        client.close()
