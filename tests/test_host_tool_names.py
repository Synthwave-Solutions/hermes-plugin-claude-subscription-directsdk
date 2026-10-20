"""Decode exact advertised names; pass other nonempty names to the host so it can answer them."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk

@pytest.mark.parametrize('returned,expected', [('mcp__hermes__read_file','read_file'), ('read_file','read_file')])
def test_exact_current_tool_name(returned,expected):
    assert directsdk.host_tool_name(returned, {'read_file'}) == expected

@pytest.mark.parametrize('returned,expected', [
    ('old_tool', 'old_tool'),
    ('mcp__hermes__old_tool', 'old_tool'),
    ('mcp__hermes__computer_use', 'computer_use'),
    ('mcp__other__read_file', 'mcp__other__read_file'),
    ('Read_File', 'Read_File'),
    ('mcp__hermes__mcp__hermes__read_file', 'mcp__hermes__read_file'),
])
def test_unknown_tool_is_passed_to_host(returned, expected):
    assert directsdk.host_tool_name(returned, {'read_file'}) == expected

def test_empty_inventory_passes_name_through():
    assert directsdk.host_tool_name('read_file', set()) == 'read_file'

@pytest.mark.parametrize('returned', ['', '   ', 'mcp__hermes__', None, 42])
def test_empty_or_non_string_name_is_rejected(returned):
    with pytest.raises(RuntimeError, match='outside the current host inventory'):
        directsdk.host_tool_name(returned, {'read_file'})

def test_canonical_prefix_takes_precedence():
    assert directsdk.host_tool_name('mcp__hermes__read_file', {'read_file','mcp__hermes__read_file'}) == 'read_file'

@pytest.mark.parametrize('streaming', [False, True])
def test_bare_name_round_trip_through_native_transport(tmp_path, streaming):
    import sys, os
    from test_directsdk import FAKE, Contract
    script = tmp_path / 'native.py'
    script.write_text(FAKE.replace("'name':'mcp__hermes__probe'", "'name':'probe'"))
    client = directsdk.Client(command=[sys.executable, str(script)], env={'PATH':os.environ['PATH'], 'HOME':str(tmp_path)})
    try:
        request = Contract().request()
        result = client.chat.completions.create(**request, stream=streaming)
        if streaming:
            chunks = list(result)
            calls = chunks[-1].choices[0].delta.tool_calls
        else:
            calls = result.choices[0].message.tool_calls
        assert calls[0].function.name == 'probe'
    finally:
        client.close()
