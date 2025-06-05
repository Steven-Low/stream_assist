"""Minimal class to manage audio processor device state."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from homeassistant.core import callback

@dataclass
class StreamAssistDevice:
    """Minimal audio processor device class with is_active and is_muted."""

    processor_id: str
    device_id: str
    is_active: bool = False
    is_muted: bool = False

    _is_active_listener: Callable[[], None] | None = None
    _is_muted_listener: Callable[[], None] | None = None


    @callback
    def set_is_active(self, active: bool) -> None:
        """Set active state."""
        if active != self.is_active:
            self.is_active = active
            if self._is_active_listener is not None:
                self._is_active_listener()

    @callback
    def set_is_muted(self, muted: bool) -> None:
        """Set muted state."""
        if muted != self.is_muted:
            self.is_muted = muted
            if self._is_muted_listener is not None:
                self._is_muted_listener()


    @callback
    def set_is_active_listener(self, is_active_listener: Callable[[], None]) -> None:
        """Listen for updates to is_active."""
        self._is_active_listener = is_active_listener

    @callback
    def set_is_muted_listener(self, is_muted_listener: Callable[[], None]) -> None:
        """Listen for updates to muted status."""
        self._is_muted_listener = is_muted_listener