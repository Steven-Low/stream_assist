import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback, HomeAssistant
from homeassistant.const import CONF_API_KEY

from .const import DOMAIN, CONF_STREAM_URL, CONF_CHAT_MODEL, CONF_HOST

from websockets.asyncio.client import connect
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



class StreamAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
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
                return self.async_create_entry(title="Stream Assist", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_STREAM_URL): str,
                vol.Required(CONF_API_KEY): str
            }),
            errors=errors
        )

