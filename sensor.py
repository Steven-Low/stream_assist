"""Binary sensor for Gemini Live."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback


from .const import DOMAIN
from .entity import GeminiLiveEntity

if TYPE_CHECKING:
    from .models import DomainDataItem


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensor entities."""
    item: DomainDataItem = hass.data[DOMAIN][config_entry.entry_id]

    # Setup is only forwarded for satellites
    assert item.device is not None

    sensor = GeminiLiveResponseSensor(item.device)
    async_add_entities([sensor])
    hass.data.setdefault(DOMAIN, {})["response_text"] = sensor


class GeminiLiveResponseSensor(GeminiLiveEntity, SensorEntity):
    """Entity to represent Assist is in progress for satellite."""

    entity_description = SensorEntityDescription(
        key="response_text",
        translation_key="response_text",

    )
    _attr_native_value = None

    def update_text(self, text):
        self._attr_native_value = text
        self.async_write_ha_state()

    # async def async_added_to_hass(self) -> None:
    #     """Call when entity about to be added to hass."""
    #     await super().async_added_to_hass()

    #     self._device.set_is_active_listener(self._is_active_changed)

    # @callback
    # def _is_active_changed(self) -> None:
    #     """Call when active state changed."""
    #     self._attr_is_on = self._device.is_active
    #     self.async_write_ha_state()