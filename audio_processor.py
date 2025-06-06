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
        self._audio_capture_executor_future = None

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
        if self.device.is_active:
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
                    audio_buffer.extend(pcm_data)

                    # Process full chunks from the buffer
                    while len(audio_buffer) >= CHUNK_BYTE_SIZE:

                        chunk = bytes(audio_buffer[:CHUNK_BYTE_SIZE])
                        del audio_buffer[:CHUNK_BYTE_SIZE]

                        future = asyncio.run_coroutine_threadsafe(
                            self._audio_queue.put({"data": chunk,"mime_type": "audio/pcm",}),
                            self.hass.loop
                        )

                        try:
                            future.result(timeout=2) # Wait for put to complete within 2s
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

                if not self.stream_url:
                    _LOGGER.info("No stream URL, skipping audio capture task.")
                    return
                else:
                    open_success = await self.hass.async_add_executor_job(
                        self.open, self.stream_url
                    )
                    if not open_success:
                        _LOGGER.error("Failed to open AV stream.")
                        await self.async_stop(called_from_start_failure=True)
                        return

                _LOGGER.info("Audio Task: Launching blocking audio capture in executor.")
                self._audio_capture_executor_future = self.hass.loop.run_in_executor(
                    None, self._audio_capture_loop   # synchronous method
                )

                await self._audio_capture_executor_future
                _LOGGER.info("Audio Task: Blocking audio capture executor job completed.")

                while self.device.is_muted:
                    if self._audio_capture_executor_future and not self._audio_capture_executor_future.done():
                        self._audio_capture_executor_future.cancel()
                    await self.on_muted()

        except Exception as e:
            _LOGGER.error(f"Audio Task: Error in audio capture executor job ~ {e}", exc_info=True)


    async def _receive_audio_task(self) -> None:
        """Task to process audio responses from Gemini."""
        if not self.session:
            return

        _LOGGER.info("Starting audio receive task")
        try:
            while self.is_running and self.session:
                async for response in self.session.receive():
                    if not self.is_running: break
                    if audio_data := response.data:
                        _LOGGER.info("Received audio: ignored now")
                    if text := response.text:
                        _LOGGER.info(f"Received text: {text}")
        except asyncio.CancelledError:
            if self.is_running:
                _LOGGER.info("Audio receive task cancelled")
        finally:
            _LOGGER.info("Receive response task finished.")


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
                    lambda: genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
                )

                self._session_manager = client.aio.live.connect(model=CONF_CHAT_MODEL, config=CONFIG_RESPONSE)
                self.session = await self._session_manager.__aenter__()

                # Check stream availability
                if not self.stream_url:
                   _LOGGER.error("Async Start: No stream url provided")
                   raise

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

        _LOGGER.info("StreamAssistSatellite stopped.")
        self.is_running = False


    async def run(self):
        _LOGGER.info(f"StreamAssistSatellite.run() called. Device active: {self.device.is_active}")
        if self.device.is_active:
            await self.async_start() # Add await here
        else:
            await self.async_stop()  # Add await here