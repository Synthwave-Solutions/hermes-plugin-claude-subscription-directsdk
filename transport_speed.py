"""Lossless request transport and request-owned speculative connection setup."""
import gzip
import io
import os
import threading
import time


def enabled(name):
    return os.environ.get(name, '1').strip().lower() not in ('0', 'false', 'no', 'off')


def decode_request(data, encoding, limit=32 * 1024 * 1024):
    encoding = (encoding or 'identity').strip().lower()
    if encoding == 'gzip':
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(limit + 1)
    elif encoding != 'identity':
        raise ValueError('unsupported request encoding')
    if len(data) > limit:
        raise ValueError('request exceeds decompressed size limit')
    return data


def encode_request(data, *, enabled):
    if enabled and len(data) >= 32 * 1024:
        compressed = gzip.compress(data, compresslevel=1, mtime=0)
        if len(compressed) < len(data):
            return compressed, 'gzip'
    return data, None


class PreparedConnection:
    """One connection, no HTTP request until take(); close owns cancellation.

    DNS/connect is bounded by the factory's timeout. A cancelled setup worker
    closes its connection as soon as connect returns, even if the caller left.
    No credentials, cross-account pools, retries, or model calls are involved.
    """
    def __init__(self, factory):
        self.factory = factory
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.closed = False
        self.conn = None
        self.error = None
        self.connect_ms = None
        self.connected_at = None
        self.thread = threading.Thread(target=self._run, name='directsdk-preconnect', daemon=True)
        self.thread.start()

    def _run(self):
        conn = None
        started = time.perf_counter()
        try:
            conn = self.factory()
            conn.connect()
            self.connected_at = time.monotonic()
            with self.lock:
                if not self.closed:
                    self.conn, conn = conn, None
        except Exception as exc:
            self.error = exc
        finally:
            if conn is not None:
                conn.close()
            self.connect_ms = (time.perf_counter() - started) * 1000
            self.ready.set()

    def take(self, timeout):
        if not self.ready.wait(timeout):
            self.close()
            raise TimeoutError('preconnect timed out')
        with self.lock:
            if self.closed:
                raise OSError('preconnect cancelled')
            if self.error is not None:
                raise self.error
            conn, self.conn = self.conn, None
            self.closed = True
        # A slow native startup can outlive an idle server connection. Reconnect
        # before submitting anything; never retry a POST with ambiguous outcome.
        if time.monotonic() - self.connected_at > 5:
            conn.close()
            conn = self.factory()
            try:
                conn.connect()
            except Exception:
                conn.close()
                raise
        return conn

    def close(self):
        with self.lock:
            self.closed = True
            conn, self.conn = self.conn, None
        if conn is not None:
            conn.close()
