
from dataclasses import dataclass


from .devices import GeminiLiveDevice


@dataclass
class DomainDataItem:
    """Domain data item."""

    device: GeminiLiveDevice | None = None
