import logging
import av
from av.audio.resampler import AudioResampler
from av.container.input import InputContainer
import asyncio
import os
import uuid
import io
import wave
from datetime import datetime, timedelta
from functools import partial
import google.genai as genai

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.device_registry import DeviceInfo

from .const import SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, DOMAIN, CONF_CHAT_MODEL
from .devices import StreamAssistDevice


_LOGGER = logging.getLogger(__name__)
CHUNK_BYTE_SIZE = SAMPLE_CHANNELS * 1024 * 2
CONFIG_RESPONSE = {"response_modalities": ["TEXT"]}

class StreamAssistSatellite():

    # entity_description.key = "stream_satellite"
    _attr_translation_key = "stream_satellite"
    _attr_name = None

    def __init__(self, hass: HomeAssistant, device: StreamAssistDevice, config: ConfigEntry):


        self.hass = hass
        self.device = device
        self.config = config

        self.is_running = True

        self._tasks: set[asyncio.Task] | None = set()
        self.session = None
        self.container: InputContainer | None = None
        self.audio_resampler: AudioResampler = None
        self._stop_event = asyncio.Event()
        self._muted_changed_event = asyncio.Event()
        self._power_changed_event = asyncio.Event()

        self.stream_url = self.config.data.get("stream_url")
        self.api_key = self.config.data.get("api_key")

        self._session_manager = None
        self._session_lock = asyncio.Lock()

        # Communication queues
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        # self.input_queue = asyncio.Queue(maxsize=10)

        # Listener
        self.device.set_is_muted_listener(self._muted_changed)
        self.device.set_is_active_listener(self._power_changed)


    def _muted_changed(self) -> None:
        """Run when device muted status changes."""
        if self.device.is_muted:
            # Cancel any running pipeline
            self._audio_queue.put_nowait(None)
            _LOGGER.info("<====== Audio Pipeline muted =====>")

            self._muted_changed_event.set()
        else:
            self._muted_changed_event.clear()
            _LOGGER.info("<====== Audio Pipeline unmuted =====>")

    def _power_changed(self) -> None:
        """Run when device power status changes."""
        if self.device.is_active:
            self._power_changed_event.set()
            self.is_running = True
            _LOGGER.info("<====== Assist Power On =====>")
            self.hass.async_create_task(self.async_start())
        else:
            self._power_changed_event.clear()
            self.is_running = False
            _LOGGER.info("<====== Assist Power Off =====>")
            self.hass.async_create_task(self.async_stop())



    def open(self, file: str, **kwargs):
        """Open the audio stream using PyAV."""
        _LOGGER.info("Opening stream: %s", file)

        if "options" not in kwargs:
            kwargs["options"] = {
                "fflags": "nobuffer",
                "flags": "low_delay",
                "timeout": "5000000",
            }

        if file.startswith("rtsp"):
            kwargs["options"].update({
                "rtsp_flags": "prefer_tcp",
                "allowed_media_types": "audio"
            })

        kwargs.setdefault("timeout", 5)
        self.container = av.open(file, **kwargs)
        self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)



    @staticmethod
    async def convert_raw_to_wav(audio_data: bytes) -> bytes:
        """Convert PCM raw audio to WAV bytes in memory."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(SAMPLE_CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        return buffer.getvalue()


    async def _send_realtime_task(self):
        _LOGGER.info("Sending realtime audio task")
        while True:
            msg = await self._audio_queue.get()
            await self.session.send(input=msg)

    async def _audio_capture_task(self) -> None:
        """Capture audio from RTSP stream and process for Gemini."""
        if not self.container:
            _LOGGER.error("RTSP container not initialized")
            return

        try:
            _LOGGER.info("Starting RTSP audio capture task")
            audio_buffer = bytearray()

            for frame in self.container.decode(audio=0):
                resampled_frames = self.audio_resampler.resample(frame)

                for resampled_frame in resampled_frames:
                    # Convert frame to raw PCM bytes
                    pcm_data = resampled_frame.to_ndarray().tobytes()
                    audio_buffer.extend(pcm_data)

                    # Process full chunks from the buffer
                    while len(audio_buffer) >= CHUNK_BYTE_SIZE:
                        # Extract one chunk
                        chunk = bytes(audio_buffer[:CHUNK_BYTE_SIZE])
                        del audio_buffer[:CHUNK_BYTE_SIZE]

                        # Queue chunk using same format as PyAudio version
                        await self._audio_queue.put({
                            "data": chunk,
                            "mime_type": "audio/pcm",
                        })

        except Exception as e:
            _LOGGER.error(f"RTSP task failed: {e}", exc_info=True)




    async def _receive_audio_task(self) -> None:
        """Task to process audio responses from Gemini."""
        if not self.session:
            return

        _LOGGER.info("Starting audio receive task")
        try:
            while True:
                async for response in self.session.receive():
                    if audio_data := response.data:
                        _LOGGER.info("Received audio: ignored now")
                    if text := response.text:
                        _LOGGER.info(f"Received text: {text}")
        except asyncio.CancelledError:
            _LOGGER.info("Audio receive task cancelled")


    async def async_start(self):
        """Start Gemini Live session and all background tasks."""

        async with self._session_lock:
            if self.session:
                return

            try:
                # client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
                client = await self.hass.async_add_executor_job(
                    lambda: genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
                )

                self._session_manager = client.aio.live.connect(model=CONF_CHAT_MODEL, config=CONFIG_RESPONSE)
                self.session = await self._session_manager.__aenter__()


                # Initialize RTSP capture if URL provided
                if self.stream_url:
                    # self.open(self.stream_url)
                    await self.hass.async_add_executor_job(partial(self.open, self.stream_url))

                # Create background tasks
                tasks = [
                    self.hass.async_create_background_task(
                        task,
                        f"gemini_live_{name}",
                        eager_start=True
                    )
                    for name, task in [
                        ("rtsp_audio", self._audio_capture_task()),
                        # ("video_input", self._video_input_task()),
                        ("realtime_send", self._send_realtime_task()),
                        ("receive_audio", self._receive_audio_task()),
                        # Removed play_audio_task since we'll play through Gemini
                    ]
                ]

                self._tasks = set(tasks)
                _LOGGER.info("Gemini Live API tasks started")

            except Exception as e:
                _LOGGER.error(f"Error starting Gemini Live: {e}", exc_info=True)
                await self.async_stop()
                raise


    async def async_stop(self) -> None:
        """Stop all tasks and clean up resources."""
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close session if exists
        if self.session:
            await self.session.close()
            self.session = None

        # Close RTSP stream
        if self.container:
            await self.hass.async_add_executor_job(self.container.close)
            self.container = None

        _LOGGER.info("Gemini Live API stopped")