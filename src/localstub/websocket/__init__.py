"""WebSocket protocol support (RFC 6455, RFC 7692).

The codec, handshake and records do no I/O; ``session`` drives one
server-side session and ``relay`` passes a session through the TLS
proxy's forward mode.  This package imports only ``localstub.http``
and ``localstub.clock``; the server wires it to a connection.

This package is internal; import public names from ``localstub``.
"""
