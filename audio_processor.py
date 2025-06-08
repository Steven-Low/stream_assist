import logging
import av
from av.audio.resampler import AudioResampler
from av.container.input import InputContainer
import asyncio
import os
import uuid
import io
import wave
import base64
from datetime import datetime, timedelta
from functools import partial
import google.genai as genai
from google.genai import types
from collections import deque

from homeassistant.const import ATTR_ENTITY_ID, CONF_EXTERNAL_URL
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.network import get_url
from homeassistant.components.media_player import (
    DOMAIN as MEDIA_PLAYER_DOMAIN,
    SERVICE_PLAY_MEDIA,
)

from .const import SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, DOMAIN, CONF_CHAT_MODEL, CONF_MEDIA_PLAYER
from .devices import GeminiLiveDevice


_LOGGER = logging.getLogger(__name__)
CHUNK_BYTE_SIZE = SAMPLE_CHANNELS * 1024 * 2
API_VERSION = "v1beta"
CONFIG_RESPONSE = {
    "response_modalities": ["AUDIO"],
}
# To enable proactivity, set api_version to v1alpha in client
# CONFIG = types.LiveConnectConfig(
#     response_modalities=["AUDIO"],
#     proactivity={'proactive_audio': True}
# )


class GeminiLiveSatellite():

    # entity_description.key = "stream_satellite"
    _attr_translation_key = "stream_satellite"
    _attr_name = None

    def __init__(self, hass: HomeAssistant, device: GeminiLiveDevice, config: ConfigEntry):

        # Core
        self.hass = hass
        self.device = device
        self.config = config
        self.stream_url = None
        self.api_key = self.config.data.get("api_key")
        self.media_player_entity_id: str | None = None
        self.external_url = self.config.options.get(CONF_EXTERNAL_URL)

        # State
        self.is_running = True

        # Async / Concurrency
        self._stop_event = asyncio.Event()
        self._muted_changed_event = asyncio.Event()
        self._power_changed_event = asyncio.Event()
        self._session_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        self._current_audio_buffer = bytearray()

        # Streaming
        self.session = None
        self.container: InputContainer | None = None
        self.audio_resampler: AudioResampler = None
        self._audio_capture_executor_future = None

        # Internal session manager
        self._session_manager = None

        # Listeners
        self.device.set_is_muted_listener(self._muted_changed)
        self.device.set_is_power_listener(self._power_changed)

        # self.config.add_update_listener(self._async_options_updated_callback)


    async def _async_options_updated_callback(self, hass:HomeAssistant, entry: ConfigEntry) -> None:
        self._update_media_player_config()

    def _update_media_player_config(self) -> None:
        self.media_player_entity_id = self.config.options.get(CONF_MEDIA_PLAYER)
        _LOGGER.info(f"Media player for audio output updated to: {self.media_player_entity_id}")

    def _muted_changed(self) -> None:
        """Run when device muted status changes."""

        if self.device.is_muted:
            self._audio_queue.put_nowait(None)
            _LOGGER.info("<====== Audio Pipeline muted =====>")
        else:
            _LOGGER.info("<====== Audio Pipeline unmuted =====>")

        self._muted_changed_event.set()
        self._muted_changed_event.clear()

    async def on_muted(self) -> None:
        """Block until device may be unmuted again."""
        await self._muted_changed_event.wait()

    def _power_changed(self) -> None:
        """Run when device power status changes."""
        if self.device.is_power:
            if not self.is_running:
                self._power_changed_event.set()
                self.is_running = True
                self.hass.async_create_task(self.async_start())
                _LOGGER.info("<====== Assist Power On =====>")
            else:
                _LOGGER.info("<====== Assist Power On: Already Running =====>")
        else:
            if self.is_running:
                self._power_changed_event.clear()
                self.is_running = False
                self.hass.async_create_task(self.async_stop())
                _LOGGER.info("<====== Assist Power Off =====>")
            else:
                _LOGGER.info("<====== Assist Power Off: Already Stopped =====>")


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
        try:
            self.container = av.open(file, **kwargs)
            self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            return True
        except Exception as e:
            self.container = None
            self.audio_resampler = None
            _LOGGER.error(f"Failed to open AV stream: {e}", exc_info=True)
            return False


    async def _send_realtime_task(self):
        _LOGGER.info("Starting send_realtime_task")
        try:
            while True: # Loop until explicitly broken or task is cancelled
                if not self.is_running or not self.session:
                    _LOGGER.info("Send task: Not running or session invalid. Exiting.")
                    break

                while self.device.is_muted:
                    _LOGGER.info("Send task: Muted. Waiting for unmute or stop.")
                    await self.on_muted()

                try:
                    msg = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue # No data, loop and re-check conditions (is_running, muted, etc.)

                if msg is None:
                    self._audio_queue.task_done()
                    _LOGGER.info("Send task: Received sentinel (None) from audio queue. Assuming stream end.")
                    continue # Wait for new audio data or for the task to be cancelled.

                if not self.device.is_muted:
                    await self.session.send(input=msg)
                else:
                    _LOGGER.info("Send task: Muted just before send, discarding message.")

                self._audio_queue.task_done()

        except asyncio.CancelledError:
            _LOGGER.info("Send_realtime_task cancelled.")
            # Propagate if needed, or just log if cancellation is part of normal shutdown.
        except Exception as e:
            _LOGGER.error(f"Error in send_realtime_task: {e}", exc_info=True)
        finally:
            _LOGGER.info("Send_realtime_task finished.")


    def _audio_capture_loop(self) -> None:
        """Capture audio from RTSP stream and process for Gemini."""
        if not self.container:
            _LOGGER.error("RTSP container not initialized")
            return

        try:
            _LOGGER.info("Starting RTSP audio capture task")

            audio_buffer = bytearray()

            for frame in self.container.decode(audio=0):
                if self._stop_event.is_set() or not self.is_running:
                    _LOGGER.info("Stopping blocking audio capture loop due to muted/stop signal.")
                    break

                resampled_frames = self.audio_resampler.resample(frame)

                for resampled_frame in resampled_frames:
                    # Convert frame to raw PCM bytes
                    pcm_data = resampled_frame.to_ndarray().tobytes()

                    # audio_buffer.extend(pcm_data)

                    # # Process full chunks from the buffer
                    # while len(audio_buffer) >= CHUNK_BYTE_SIZE:

                    #     chunk = bytes(audio_buffer[:CHUNK_BYTE_SIZE])
                    #     del audio_buffer[:CHUNK_BYTE_SIZE]

                    future = asyncio.run_coroutine_threadsafe(
                        self._audio_queue.put({"data": pcm_data,"mime_type": "audio/pcm",}),
                        self.hass.loop
                    )

                    try:
                        future.result(timeout=3) # Wait for put to complete within 2s
                    except asyncio.QueueFull:
                        _LOGGER.warning("Audio Capture Loop: Audio queue full dropping audio chunk")
                    except asyncio.TimeoutError:
                        _LOGGER.warning("Audio Capture Loop: Timeout putting audio chunk on queue.")
                        return
                    except Exception as e_put:
                        _LOGGER.error(f"Audio Capture Loop: Error putting audio chunk on queue: {e_put}")
                        return

                if self._stop_event.is_set() or not self.is_running: break

        except Exception as e_decode:
            if self.is_running and not self._stop_event.is_set(): # Log only if not intentional stop
                self._stop_event.set()
                _LOGGER.error(f"Audio Capture Loop: RTSP decode/resample loop failed {e_decode}", exc_info=True)
        finally:
            _LOGGER.info("Audio Capture Loop: Blocking RTSP loop finished.")
            self.container.close()


    async def _audio_capture_task_launcher(self) -> None:
        """Async task to launch the blocking audio capture loop in an executor."""
        try:
            while True:
                if self._stop_event.is_set():
                    _LOGGER.info("Audio Task: Stopped.")
                    return

                self.stream_url = self.config.options.get("stream_url", self.config.data.get("stream_url"))
                if not self.stream_url:
                    _LOGGER.info("No stream URL, skipping audio capture task.")
                    return
                else:
                    open_success = await self.hass.async_add_executor_job(
                        self.open, self.stream_url
                    )
                    if not open_success:
                        _LOGGER.error("Failed to open AV stream.")
                        self._stop_event.set()
                        self.is_running = False
                        self.device.set_is_active(False)
                        return
                    else:
                        self.device.set_is_active(True)

                _LOGGER.info("Audio Task: Launching blocking audio capture in executor.")
                self._audio_capture_executor_future = self.hass.loop.run_in_executor(
                    None, self._audio_capture_loop   # synchronous method
                )

                await self._audio_capture_executor_future
                _LOGGER.info("Audio Task: Blocking audio capture executor job completed.")

                if self.device.is_muted:
                    self.device.set_is_active(False)

                while self.device.is_muted:
                    if self._audio_capture_executor_future and not self._audio_capture_executor_future.done():
                        self._audio_capture_executor_future.cancel()
                    await self.on_muted()

        except Exception as e:
            _LOGGER.error(f"Audio Task: Error in audio capture executor job ~ {e}", exc_info=True)
            raise

    async def _receive_audio_task(self) -> None:
        """Task to process audio responses from Gemini."""
        if not self.session:
            return

        self._update_media_player_config()

        _LOGGER.info("Starting audio receive task")
        sentence_buffer = ""
        sentence_endings = (".", "!", "?")
        sentence_queue = deque(maxlen=1)  # store up to 3 lines
        self._current_audio_buffer.clear()

        try:
            while self.is_running and self.session:
                async for response in self.session.receive():
                    if not self.is_running:
                        break

                    if audio_data := response.data:
                        self._current_audio_buffer.extend(audio_data)


                if self._current_audio_buffer:
                    await self._play_audio(bytes(self._current_audio_buffer))
                    self._current_audio_buffer.clear()
                    _LOGGER.info("finished playing")

                    # if text := response.text:
                    #     sentence_buffer += text
                    #     _LOGGER.info(f"{text}")

                    #     if sentence_buffer.strip().endswith(sentence_endings):
                    #         final_sentence = sentence_buffer.strip()
                    #         sentence_queue.append(final_sentence)
                    #         sentence_buffer = ""  # clear buffer

                    #         # Join last 3 sentences with newlines
                    #         full_output = "\n".join(sentence_queue)

                    #         if sensor := self.hass.data[DOMAIN].get("response_text"):
                    #             sensor.update_text(full_output)

        except asyncio.CancelledError:
            if self.is_running:
                _LOGGER.info("Audio receive task cancelled")
        except Exception as e:
            _LOGGER.info(f"Receive Task: {e}")
        finally:
            _LOGGER.info("Receive response task finished.")

    async def _play_audio_task(self) -> None:
        pass

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

    async def _play_audio(self, audio_bytes: bytes) -> None:
        """Plays the accumulated audio data on the configured media_player."""
        if not self.media_player_entity_id:
            _LOGGER.warning("No media_player_entity_id configured. Cannot play audio.")
            return
        if not audio_bytes:
            _LOGGER.debug("No audio data in buffer to play.")
            return

        _LOGGER.info(f"Attempting to play {len(audio_bytes)} bytes of audio on: {self.media_player_entity_id}")

        try:
            wav_bytes = await self.convert_raw_to_wav(audio_bytes)

            output_dir = self.hass.config.path("www")  # <config_dir>/www
            file_id = f"gemini_tts_{uuid.uuid4().hex}.wav"
            file_path_on_disk = os.path.join(output_dir, file_id)

            def save_wav_to_file_sync(path: str, data: bytes):
                with open(path, "wb") as f:
                    f.write(data)

            await self.hass.async_add_executor_job(
                partial(save_wav_to_file_sync, path=file_path_on_disk, data=wav_bytes)
            )
            _LOGGER.info(f"Audio saved to temporary file: {file_path_on_disk}")

            media_url_path = f"/local/{file_id}"
            full_media_url = f"{self.external_url}{media_url_path}" #get_url(self.hass, prefer_external=False)

            _LOGGER.info(f"Playing audio from URL: {full_media_url}") # Or media_url_path

            media_content_id = full_media_url
            media_content_type = "music" # Or MediaType.AUDIO

            await self.hass.services.async_call(
                MEDIA_PLAYER_DOMAIN,
                SERVICE_PLAY_MEDIA,
                {
                    ATTR_ENTITY_ID: self.media_player_entity_id,
                    "media_content_id": media_content_id,
                    "media_content_type": media_content_type,
                    # Consider "enqueue: replace" if you want to interrupt previous playback
                    # "enqueue": "replace",
                },
                blocking=False, # Usually False for TTS-like playback
            )
            _LOGGER.debug(f"Play_media called for {self.media_player_entity_id}")

        except wave.Error as e:
            _LOGGER.error(f"Error creating WAV data for media player: {e}", exc_info=True)
        except Exception as e:
            _LOGGER.error(f"Error playing audio on media_player: {e}", exc_info=True)


    async def async_start(self):
        """Start Gemini Live session and all background tasks."""

        async with self._session_lock:
            if self.session:
                return

            self.is_running = True
            self._stop_event.clear()
            self._muted_changed_event.clear()

            try:
                client = await self.hass.async_add_executor_job(
                    lambda: genai.Client(api_key=self.api_key, http_options={"api_version": API_VERSION})
                )

                self._session_manager = client.aio.live.connect(model=CONF_CHAT_MODEL, config=CONFIG_RESPONSE)
                self.session = await self._session_manager.__aenter__()

                # if self.session:
                #     self.device.set_is_active(True)

                # Create background tasks
                task_configs = []
                task_configs.extend([
                    ("audio_capture_launcher", self._audio_capture_task_launcher()),
                    ("realtime_send", self._send_realtime_task()),
                    ("receive", self._receive_audio_task())
                ])
                for name, coro in task_configs:
                    task = self.hass.async_create_background_task(
                        coro,
                        f"gemini_live_{name}",
                        eager_start=True
                    )
                    self._tasks.add(task)
                    _LOGGER.info(f"Creating background task: gemini_live_{name}")


            except Exception as e:
                _LOGGER.error(f"Error starting Gemini Live: {e}", exc_info=True)
                await self.async_stop(called_from_start_failure=True)


    async def async_stop(self, called_from_start_failure=False) -> None:
        _LOGGER.info(f"async_stop called. is_running: {self.is_running}, called_from_start_failure: {called_from_start_failure}")

        if not called_from_start_failure:
            self.is_running = False
        self._stop_event.set() # Signal all tasks, especially blocking ones

        if self._audio_capture_executor_future and not self._audio_capture_executor_future.done():
            self._audio_capture_executor_future.cancel()
            try:
                await asyncio.wait_for(self._audio_capture_executor_future, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                _LOGGER.info("Audio capture executor future did not complete cleanly after cancel request.")
            except Exception as e_fut:
                _LOGGER.warning(f"Exception waiting for audio capture executor future: {e_fut}")
        self._audio_capture_executor_future = None


        # Cancel all asyncio background tasks
        if self._tasks:
            _LOGGER.info(f"Cancelling {len(self._tasks)} background tasks...")
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=5.0)
                _LOGGER.info("All background tasks gathered after cancellation.")
            except asyncio.TimeoutError:
                _LOGGER.warning("Timeout waiting for background tasks to cancel. Some tasks may still be running.")
            except Exception as e_gather:
                _LOGGER.error(f"Error gathering tasks during stop: {e_gather}", exc_info=True)
            self._tasks.clear()


        # Close session if exists, using the session manager for proper cleanup
        if self._session_manager:
            _LOGGER.info("Closing Gemini session using session manager...")
            try:
                await self._session_manager.__aexit__(None, None, None)
                _LOGGER.info("Gemini session closed.")
            except Exception as e_session:
                _LOGGER.error(f"Error closing Gemini session: {e_session}", exc_info=True)
            finally:
                self._session_manager = None
                self.session = None
                self.device.set_is_active(False)

        # Close RTSP stream container (blocking call)
        if self.container:
            _LOGGER.info("Closing AV container...")
            try:
                await self.hass.async_add_executor_job(self.container.close)
                _LOGGER.info("AV container closed.")
            except Exception as e_container:
                _LOGGER.error(f"Error closing AV container: {e_container}", exc_info=True)
            finally:
                self.container = None
                self.audio_resampler = None # Resampler is tied to container's streams

        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        _LOGGER.info("Audio queue cleared.")

        _LOGGER.info("GeminiLiveSatellite stopped.")
        self.is_running = False


    async def run(self):
        _LOGGER.info(f"StreamAssistSatellite.run() called. Device active: {self.device.is_power}")
        if self.device.is_power:
            await self.async_start() # Add await here
        else:
            await self.async_stop()  # Add await here