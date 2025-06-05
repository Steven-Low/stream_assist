
from dataclasses import dataclass


from .devices import StreamAssistDevice


@dataclass
class DomainDataItem:
    """Domain data item."""

    device: StreamAssistDevice | None = None
