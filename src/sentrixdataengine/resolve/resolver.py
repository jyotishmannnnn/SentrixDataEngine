"""Resolver registry + URI helpers (scheme-agnostic dispatch)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..contracts import PayloadResolver


def split_scheme(uri: str) -> Tuple[str, str]:
    """Return (scheme, rest) for ``scheme://rest``; rest keeps any ``#fragment``."""
    if "://" not in uri:
        raise ValueError(f"not a payload URI: {uri!r}")
    scheme, rest = uri.split("://", 1)
    return scheme, rest


def strip_fragment(uri: str) -> str:
    """Drop a trailing ``#fragment`` (the per-row addressing) → stream base URI."""
    return uri.split("#", 1)[0]


class ResolverRegistry:
    """Dispatches a stream base URI to the resolver that supports its scheme."""

    def __init__(self) -> None:
        self._resolvers: list[PayloadResolver] = []

    def register(self, resolver: PayloadResolver) -> "ResolverRegistry":
        self._resolvers.append(resolver)
        return self

    def for_uri(self, uri: str) -> PayloadResolver:
        scheme, _ = split_scheme(uri)
        for r in self._resolvers:
            if r.supports(scheme):
                return r
        raise ValueError(
            f"no resolver registered for scheme {scheme!r} "
            f"(uri={uri!r}); registered schemes via {len(self._resolvers)} resolver(s)")

    def resolve_stream(self, base_uri: str, payload_kind: str,
                       payload_shape: tuple[int, ...] | None) -> np.ndarray:
        return self.for_uri(base_uri).resolve_stream(
            strip_fragment(base_uri), payload_kind, payload_shape)


def default_registry() -> ResolverRegistry:
    """Registry with the shipped schemes (parquet + file, mcap)."""
    from .mcap_resolver import McapPayloadResolver
    from .parquet_resolver import ParquetPayloadResolver
    return (ResolverRegistry()
            .register(ParquetPayloadResolver())
            .register(McapPayloadResolver()))
