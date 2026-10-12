"""In-process network kill switch.

n0face never needs the network at runtime, so after startup we make any attempt to open
an IP socket or resolve a hostname from Python raise. This catches accidental telemetry from
our own code or Python dependencies.

Limitation (documented in docs/PRIVACY.md): native code that calls the OS socket API directly
bypasses this. Use an OS firewall rule for a hard guarantee.
"""

import socket

_BLOCKED_FAMILIES = {socket.AF_INET, socket.AF_INET6, -1}


class NetworkDisabledError(PermissionError):
    pass


def lock_network():
    """Disable Python-level networking for this process. Returns a function that undoes it."""
    original_socket = socket.socket
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection

    class GuardedSocket(original_socket):
        def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
            if fileno is None and family in _BLOCKED_FAMILIES:
                raise NetworkDisabledError("n0face: network access is disabled")
            super().__init__(family, type, proto, fileno)

    def blocked(*_args, **_kwargs):
        raise NetworkDisabledError("n0face: network access is disabled")

    socket.socket = GuardedSocket
    socket.getaddrinfo = blocked
    socket.create_connection = blocked

    def unlock():
        socket.socket = original_socket
        socket.getaddrinfo = original_getaddrinfo
        socket.create_connection = original_create_connection

    return unlock
