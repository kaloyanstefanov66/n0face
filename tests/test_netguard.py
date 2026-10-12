import socket

import pytest

from n0face.netguard import NetworkDisabledError, lock_network


def test_lock_blocks_ip_sockets_and_dns_then_unlocks():
    unlock = lock_network()
    try:
        with pytest.raises(NetworkDisabledError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkDisabledError):
            socket.socket()
        with pytest.raises(NetworkDisabledError):
            socket.getaddrinfo("example.com", 443)
        with pytest.raises(NetworkDisabledError):
            socket.create_connection(("127.0.0.1", 9))
    finally:
        unlock()
    socket.socket().close()


def test_urllib_fails_while_locked():
    import urllib.request

    unlock = lock_network()
    try:
        with pytest.raises(OSError):
            urllib.request.urlopen("http://example.com", timeout=1)
    finally:
        unlock()
