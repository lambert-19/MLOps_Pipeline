# Starlette's TestClient imports httpx2 for type checking, but this project
# only installs httpx (which Starlette falls back to at runtime). Alias it.
from httpx import *
from httpx import USE_CLIENT_DEFAULT as USE_CLIENT_DEFAULT

from . import _client as _client
from . import _types as _types
