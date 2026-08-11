"""URI parsing utilities for HTTP forward proxy support."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class ParsedURI:
    """Parsed components of an absolute-form URI.

    Used for forward proxy requests where the client sends the full URL
    in the request line (e.g., GET http://example.com/path HTTP/1.1).
    """

    scheme: str
    host: str
    port: int
    path: str  # includes query string, e.g., "/foo?bar=1"

    @property
    def authority(self) -> str:
        """Render the Host header value for this URI.

        IPv6 literals are bracketed (``[::1]``) so the host cannot be
        confused with a port.  The port is omitted only when it is the
        default for the scheme, so ``http://example.com:443/`` keeps
        its explicit port.
        """
        host = f"[{self.host}]" if ":" in self.host else self.host
        if self.port == _DEFAULT_PORTS.get(self.scheme):
            return host
        return f"{host}:{self.port}"


def parse_absolute_uri(uri: str) -> ParsedURI | None:
    """Parse an absolute-form URI into components.

    Args:
        uri: The URI string to parse (e.g., "http://example.com:8080/path?q=1")

    Returns:
        ParsedURI with scheme, host, port, and path if the URI is
        absolute-form, or None if the URI is origin-form (starts with "/")
        or invalid.

    Examples:
        >>> parse_absolute_uri("http://example.com/path")
        ParsedURI(scheme='http', host='example.com', port=80, path='/path')

        >>> parse_absolute_uri("https://example.com:8443/api?key=val")
        ParsedURI(scheme='https', host='example.com', port=8443,
                  path='/api?key=val')

        >>> parse_absolute_uri("/path")
        None
    """
    # Origin-form URIs start with "/" - not absolute
    if uri.startswith("/"):
        return None

    # Must have a scheme for absolute-form
    if "://" not in uri:
        return None

    parsed = urlparse(uri)

    # Validate we have required components
    if not parsed.scheme or not parsed.netloc:
        return None

    # Determine default port based on scheme
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80

    # Build path with query string
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    return ParsedURI(
        scheme=parsed.scheme,
        host=parsed.hostname or "",
        port=port,
        path=path,
    )
