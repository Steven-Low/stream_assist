from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers import config_validation as cv, device_registry as dr
from homeassistant.const import Platform

import logging
from .const import DOMAIN
from .audio_processor import GeminiLiveSatellite
from .models import DomainDataItem
from .devices import GeminiLiveDevice

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)
_LOGGER = logging.getLogger(__name__)

SATELLITE_PLATFORMS = [
    Platform.SWITCH,
    Platform.BINARY_SENSOR,
    Platform.SENSOR,
]


__all__ = [
    "DOMAIN",
    "async_setup",
    "async_setup_entry",
    "async_unload_entry",
]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Initial setup of the integration."""
    return True


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> bool:
    """Directly set up the Stream Assist entity without platform forwarding."""

    item = DomainDataItem()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = item
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # Create device in the device registry
    dev_reg = dr.async_get(hass)
    processor_id = entry.entry_id
    device = dev_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, processor_id)},
        name="GeminiLiveSatellite",
        suggested_area="Bedroom",
    )

    # Store device information
    item.device = GeminiLiveDevice(
        processor_id=processor_id,
        device_id=device.id,
    )
    # Set up satellite entity, sensors, switches, etc.
    await hass.config_entries.async_forward_entry_setups(entry, SATELLITE_PLATFORMS)

    stream = GeminiLiveSatellite(hass, item.device, entry)
    await stream.run()


    return True


async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle updates to config entry options."""
    await hass.config_entries.async_reload(entry.entry_id)


# async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
#     """Unload the config entry and clean up."""
#     if entry.entry_id in hass.data.get(DOMAIN, {}):
#         del hass.data[DOMAIN][entry.entry_id]
#     return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info(f"Unloading config entry {entry.entry_id} for domain {DOMAIN}")

    # TODO: Put Satellite on Data Item
    # item: DomainDataItem | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    # if item and hasattr(item, 'satellite') and item.satellite:
    #     _LOGGER.info(f"Stopping satellite processor for entry {entry.entry_id}.")
    #     await item.satellite.async_stop()

    _LOGGER.debug(f"Unloading platforms {SATELLITE_PLATFORMS} for entry {entry.entry_id}")
    unload_ok = await hass.config_entries.async_unload_platforms(entry, SATELLITE_PLATFORMS)

    if unload_ok:
        if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
            hass.data[DOMAIN].pop(entry.entry_id)
            _LOGGER.info(f"Successfully unloaded and cleaned up data for entry {entry.entry_id}.")
        else:
            _LOGGER.warning(f"Domain data for entry {entry.entry_id} not found during unload.")
    else:
        _LOGGER.error(f"Failed to cleanly unload one or more platforms for entry {entry.entry_id}. This may cause issues on reload.")
        return False
    return True