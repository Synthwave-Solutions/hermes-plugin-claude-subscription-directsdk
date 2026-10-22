import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
import pytest
from admission import Admission
from transport_speed import encode_request, decode_request


def test_lossless_large_json_and_small_payload():
    data=json.dumps({'messages':[{'role':'user','content':'héllo tools context ' * 10000}]}).encode()
    packed, encoding=encode_request(data, enabled=True)
    assert encoding == 'gzip'
    assert len(packed) < len(data)/5
    assert gzip.decompress(packed) == data
    assert encode_request(b'{}', enabled=True) == (b'{}', None)
    assert encode_request(data, enabled=False) == (data, None)


def test_bounded_decode_and_bad_encoding():
    assert decode_request(gzip.compress(b'hello'), 'gzip') == b'hello'
    with pytest.raises(ValueError): decode_request(gzip.compress(b'x'*100), 'gzip', limit=50)
    with pytest.raises(ValueError): decode_request(b'hello', 'br')


def test_preconnect_compression_and_single_admission(monkeypatch):
    monkeypatch.setenv('HERMES_DIRECTSDK_PRECONNECT', '1')
    received=[]
    accepted=threading.Event()
    class Server(ThreadingHTTPServer):
        def get_request(self):
            result=super().get_request(); accepted.set(); return result
    class Peer(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            raw=self.rfile.read(int(self.headers['Content-Length']))
            received.append((dict(self.headers),raw))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            self.wfile.write(b'data: {"type":"ping"}\n\n'); self.wfile.flush()
    peer=Server(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    gate=Admission(f'http://127.0.0.1:{peer.server_port}',5)
    try:
        assert accepted.wait(2), 'TCP connection must precede native request'
        assert not received, 'preconnect must not submit any request'
        # Explicit local fixture only; production enables gzip only for Anthropic.
        gate.gzip_requests=True
        data=json.dumps({'messages':[{'role':'user','content':'hello '*20000}]}).encode()
        req=Request(gate.url+'/v1/messages',data=data,headers={'Content-Type':'application/json'})
        assert b'ping' in urlopen(req,timeout=3).read()
        assert len(received)==1
        assert received[0][0]['Content-Encoding']=='gzip'
        assert gzip.decompress(received[0][1])==data
        assert gate.transport_metrics['wire_bytes'] < len(data)
        assert gate.transport_metrics['first_byte_ms'] >= 0
        assert gate.transport_metrics['connect_wait_ms'] < 100
    finally:
        gate.close(); peer.shutdown(); peer.server_close(); thread.join()


def test_cancel_during_connect_releases_late_socket():
    from transport_speed import PreparedConnection
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Conn:
        def connect(self): entered.set(); release.wait(2)
        def close(self): closed.set()
    prepared=PreparedConnection(Conn)
    assert entered.wait(1)
    prepared.close()
    release.set()
    assert closed.wait(1)
    with pytest.raises(OSError): prepared.take(1)


def test_compressed_native_body_preserves_content_and_breakpoint():
    from admission import pin_message_breakpoint
    body={'messages':[{'role':'assistant','content':[{'type':'text','text':'past'}]},
                      {'role':'user','content':[{'type':'text','text':'hello'},
                                               {'type':'text','text':'native context','cache_control':{'type':'ephemeral'}}]}]}
    payload=json.dumps(body).encode()
    decoded=decode_request(gzip.compress(payload),'gzip')
    assert pin_message_breakpoint(decoded,[{'type':'text','text':'hello'}]) == pin_message_breakpoint(payload,[{'type':'text','text':'hello'}])


def test_ping_is_not_first_output():
    from admission import Capture
    capture=Capture()
    capture.feed(b'data: {"type":"ping"}\n\n')
    assert capture.first_output_at is None


def test_old_preconnect_is_refreshed_before_any_request():
    from transport_speed import PreparedConnection
    connections=[]
    class Conn:
        closed=False
        def __init__(self): connections.append(self)
        def connect(self): pass
        def close(self): self.closed=True
    prepared=PreparedConnection(Conn)
    assert prepared.ready.wait(1)
    prepared.connected_at-=10
    conn=prepared.take(1)
    try:
        assert len(connections)==2 and connections[0].closed
        assert conn is connections[1]
    finally: conn.close()
