
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback, HomeAssistant
from homeassistant.const import CONF_API_KEY, CONF_EXTERNAL_URL
from homeassistant.helpers import selector, entity_registry

from .const import (
    DOMAIN,
    CONF_STREAM_URL,
    CONF_CHAT_MODEL,
    CONF_HOST,
    CONF_MEDIA_PLAYER,
    DEFAULT_STREAM_URL
)
from websockets.asyncio.client import connect
import voluptuous as vol
from typing import Any
import logging
import json



_LOGGER = logging.getLogger(__name__)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]):
    api_key = data[CONF_API_KEY]
    uri = f"wss://{CONF_HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"

    async with connect(uri, additional_headers={"Content-Type": "application/json"}) as ws:
        setup_msg = {"setup": {"model": f"models/{CONF_CHAT_MODEL}"}}
        await ws.send(json.dumps(setup_msg))
        raw_response = await ws.recv(decode=False)
        setup_response = json.loads(raw_response.decode("ascii"))
        _LOGGER.info("API connection successful: %s", setup_response)
        ws.close()


class GeminiLiveConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Stream Assist."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""

        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_input(self.hass, user_input)
            except Exception:
                errors["base"] = "Invalid Authentication"
                _LOGGER.exception("Authentication Error")
            else:
                return self.async_create_entry(title="Gemini Live", data=user_input, options={})

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_STREAM_URL, default=DEFAULT_STREAM_URL): str,
                vol.Required(CONF_API_KEY): str
            }),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return GeminiLiveOptionsFlowHandler()


class GeminiLiveOptionsFlowHandler(OptionsFlow):
    """Handle a options flow for Google Cloud STT integration."""

    @property
    def config_entry(self):
        return self.hass.config_entries.async_get_entry(self.handler)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="Gemini Live", data=user_input)

        current_stream = self.config_entry.options.get(CONF_STREAM_URL, self.config_entry.data.get(CONF_STREAM_URL))
        # current_player = self.config_entry.options.get(CONF_MEDIA_PLAYER)

        dynamic_schema = vol.Schema({
            vol.Optional(
                CONF_STREAM_URL,
                default=current_stream
            ): str,
            vol.Optional(
                CONF_MEDIA_PLAYER,
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="media_player"),
            ),
            vol.Optional(
                CONF_EXTERNAL_URL,
                description={"suggested_value": "https://homeassistant.duckdns.com"}
            ): str
        })

        return self.async_show_form(step_id="init", data_schema=dynamic_schema)