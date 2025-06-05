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
import time # For potential sleep in blocking task

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
# from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback # Not used in this snippet
# from homeassistant.helpers.device_registry import DeviceInfo # Not used in this snippet

from .const import SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, DOMAIN, CONF_CHAT_MODEL
from .devices import StreamAssistDevice


_LOGGER = logging.getLogger(__name__)
CHUNK_BYTE_SIZE = SAMPLE_CHANNELS * 1024 * 2 # This is 4096 bytes for mono, 16-bit. At 44100Hz, this is ~46ms.
                                            # For SAMPLE_RATE = 16000, this is ~128ms.
CONFIG_RESPONSE = {"response_modalities": ["TEXT"]}

class StreamAssistSatellite():

    _attr_translation_key = "stream_satellite"
    _attr_name = None

    def __init__(self, hass: HomeAssistant, device: StreamAssistDevice, config: ConfigEntry):
        self.hass = hass
        self.device = device
        self.config = config

        self.is_running = False # Start as not running

        self._tasks: set[asyncio.Task] = set() # Initialize as empty set
        self.session = None
        self.container: InputContainer | None = None
        self.audio_resampler: AudioResampler | None = None # Initialize as None
        self._stop_event = asyncio.Event()
        self._muted_changed_event = asyncio.Event()
        self._power_changed_event = asyncio.Event() # Not strictly needed if power directly calls start/stop

        self.stream_url = self.config.data.get("stream_url")
        self.api_key = self.config.data.get("api_key")

        self._session_manager = None
        self._session_lock = asyncio.Lock()

        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=20) # Increased maxsize a bit

        self.device.set_is_muted_listener(self._muted_changed)
        self.device.set_is_active_listener(self._power_changed)
        self._audio_capture_executor_future = None


    def _muted_changed(self) -> None:
        if self.device.is_muted:
            _LOGGER.info("<====== Audio Pipeline muted =====>")
            self._muted_changed_event.set()
            # The capture loop and send task will check self.device.is_muted
            # or wait on this event.
            # Avoid putting None on queue here if send_task handles mute by pausing.
        else:
            self._muted_changed_event.clear() # Signal tasks waiting on this event
            _LOGGER.info("<====== Audio Pipeline unmuted =====>")

    def _power_changed(self) -> None:
        if self.device.is_active:
            if not self.is_running: # Only start if not already running
                _LOGGER.info("<====== Assist Power On: Starting =====>")
                self.is_running = True
                self._stop_event.clear() # Clear stop event before starting
                self.hass.async_create_task(self.async_start())
            else:
                _LOGGER.info("<====== Assist Power On: Already running =====>")
        else:
            if self.is_running: # Only stop if currently running
                _LOGGER.info("<====== Assist Power Off: Stopping =====>")
                self.is_running = False # Set this first
                # self._stop_event.set() # Signal blocking tasks to stop
                self.hass.async_create_task(self.async_stop())
            else:
                _LOGGER.info("<====== Assist Power Off: Already stopped =====>")


    def _open_av_stream_blocking(self, file: str, **kwargs):
        """Blocking function to open the audio stream using PyAV."""
        _LOGGER.info("Opening stream (blocking): %s", file)

        options = kwargs.pop("options", {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "timeout": "5000000", # 5 seconds
        })

        if file.startswith("rtsp"):
            options.update({
                "rtsp_flags": "prefer_tcp",
                "allowed_media_types": "audio"
            })
        kwargs.setdefault("timeout", 5) # av.open timeout

        try:
            self.container = av.open(file, options=options, **kwargs)
            self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            _LOGGER.info("Stream opened successfully (blocking)")
            return True
        except Exception as e:
            _LOGGER.error(f"Failed to open AV stream (blocking): {e}", exc_info=True)
            self.container = None
            self.audio_resampler = None
            return False

    @staticmethod
    async def convert_raw_to_wav(audio_data: bytes) -> bytes:
        """Convert PCM raw audio to WAV bytes in memory."""
        buffer = io.BytesIO()
        # Run wave.open in executor as it can be blocking for large data
        def _blocking_wave_write():
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(SAMPLE_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
            return buffer.getvalue()
        return await asyncio.get_event_loop().run_in_executor(None, _blocking_wave_write)


    def _blocking_audio_capture_loop(self) -> None:
        """
        Synchronous (blocking) function to capture and process audio.
        This runs in an executor thread.
        """
        if not self.container or not self.audio_resampler:
            _LOGGER.error("RTSP container or resampler not initialized for blocking capture.")
            return

        _LOGGER.info("Starting blocking RTSP audio capture loop in executor")
        audio_buffer_sync = bytearray()
        last_frame_time = time.monotonic()

        try:
            for frame in self.container.decode(audio=0):
                if self._stop_event.is_set() or not self.is_running:
                    _LOGGER.info("Stopping blocking audio capture loop due to stop signal.")
                    break

                current_time = time.monotonic()
                # _LOGGER.debug(f"Time since last frame: {current_time - last_frame_time:.3f}s")
                last_frame_time = current_time

                if self.device.is_muted:
                    # If muted, skip processing but keep stream alive.
                    # Small sleep to yield CPU if decode returns very fast with no data.
                    time.sleep(0.01)
                    continue

                resampled_frames = self.audio_resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    pcm_data = resampled_frame.to_ndarray().tobytes()
                    audio_buffer_sync.extend(pcm_data)

                    while len(audio_buffer_sync) >= CHUNK_BYTE_SIZE:
                        if self._stop_event.is_set() or not self.is_running: break # Check again
                        chunk = bytes(audio_buffer_sync[:CHUNK_BYTE_SIZE])
                        del audio_buffer_sync[:CHUNK_BYTE_SIZE]

                        # Put on the asyncio queue from the executor thread
                        future = asyncio.run_coroutine_threadsafe(
                            self._audio_queue.put({"data": chunk, "mime_type": "audio/pcm"}),
                            self.hass.loop
                        )
                        try:
                            future.result(timeout=2) # Wait for put to complete, with timeout
                        except asyncio.QueueFull: # Should not happen if queue.put handles full
                            _LOGGER.warning("Audio queue full, dropping audio chunk.")
                            # To prevent overflow, could clear some old items or just drop current
                        except asyncio.TimeoutError:
                            _LOGGER.warning("Timeout putting audio chunk on queue. Stopping capture.")
                            self._stop_event.set() # Signal stop
                            break
                        except Exception as e_put:
                            _LOGGER.error(f"Error putting audio chunk on queue: {e_put}")
                            self._stop_event.set() # Signal stop
                            break
                if self._stop_event.is_set() or not self.is_running: break

        except StopIteration:
            _LOGGER.info("RTSP stream ended (StopIteration).")
        except av.error.EOFError:
            _LOGGER.info("RTSP stream EOF.")
        except Exception as e_decode:
            if self.is_running and not self._stop_event.is_set(): # Log only if not intentional stop
                _LOGGER.error(f"RTSP decode/resample loop failed: {e_decode}", exc_info=True)
        finally:
            _LOGGER.info("Blocking RTSP audio capture loop finished.")
            # Signal end of stream to send_realtime_task
            if self.is_running: # Only put sentinel if we weren't stopped by _stop_event
                asyncio.run_coroutine_threadsafe(self._audio_queue.put(None), self.hass.loop).result(timeout=1)


    async def _audio_capture_task_launcher(self) -> None:
        """Async task to launch the blocking audio capture loop in an executor."""
        if not self.stream_url:
            _LOGGER.info("No stream URL, skipping audio capture task.")
            return

        _LOGGER.info("Launching blocking audio capture in executor.")
        # `_blocking_audio_capture_loop` is a synchronous method
        self._audio_capture_executor_future = self.hass.loop.run_in_executor(
            None, self._blocking_audio_capture_loop
        )
        try:
            await self._audio_capture_executor_future
            _LOGGER.info("Blocking audio capture executor job completed.")
        except asyncio.CancelledError:
            _LOGGER.info("Audio capture launcher task cancelled, ensuring executor future is cancelled.")
            if self._audio_capture_executor_future:
                self._audio_capture_executor_future.cancel() # Attempt to cancel if it's still running
            raise # Re-raise CancelledError
        except Exception as e:
            _LOGGER.error(f"Error in audio capture executor job: {e}", exc_info=True)


    async def _send_realtime_task(self):
        _LOGGER.info("Starting send_realtime_task")
        try:
            while True: # Loop until explicitly broken or task is cancelled
                if not self.is_running or not self.session:
                    _LOGGER.info("Send task: Not running or session invalid. Exiting.")
                    break

                if self.device.is_muted:
                    _LOGGER.info("Send task: Muted. Waiting for unmute or stop.")
                    # Wait for either unmute or stop event
                    unmuted_event_task = self.hass.async_create_task(self._muted_changed_event.wait())
                    stop_event_task = self.hass.async_create_task(self._stop_event.wait())
                    done, pending = await asyncio.wait(
                        {unmuted_event_task, stop_event_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel() # Cancel the other waiting task

                    if stop_event_task in done or self._stop_event.is_set():
                        _LOGGER.info("Send task: Stop event received while muted. Exiting.")
                        break
                    if unmuted_event_task in done:
                        self._muted_changed_event.clear() # Reset for next mute
                        _LOGGER.info("Send task: Unmuted. Resuming.")
                        # No need to drain queue if capture task respects mute
                        continue # Re-check conditions

                try:
                    msg = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue # No data, loop and re-check conditions (is_running, muted, etc.)

                if msg is None:
                    _LOGGER.info("Send task: Received sentinel (None) from audio queue. Assuming stream end.")
                    # If you want the session to end when audio ends, you might break here.
                    # Or, if the session should persist for new audio, continue.
                    # For now, let's assume it means end of this audio stream for the session.
                    # Gemini might need a specific signal for "end of user audio turn".
                    # This could be an empty send: await self.session.send(input=None) or similar.
                    # For now, just log and stop sending audio from this ended stream.
                    # If session should close, it will be handled by async_stop or external logic.
                    self._audio_queue.task_done()
                    continue # Wait for new audio data or for the task to be cancelled.

                if not self.device.is_muted: # Double check mute status
                    await self.session.send(input=msg)
                else:
                    _LOGGER.debug("Send task: Muted just before send, discarding message.")

                self._audio_queue.task_done()

        except asyncio.CancelledError:
            _LOGGER.info("Send_realtime_task cancelled.")
            # Propagate if needed, or just log if cancellation is part of normal shutdown.
        except Exception as e:
            _LOGGER.error(f"Error in send_realtime_task: {e}", exc_info=True)
        finally:
            _LOGGER.info("Send_realtime_task finished.")


    async def _receive_task(self) -> None: # Renamed from _receive_audio_task
        """Task to process responses from Gemini."""
        if not self.session:
            _LOGGER.warning("Receive task: No session available.")
            return

        _LOGGER.info("Starting receive task")
        try:
            while self.is_running and self.session:
                async for response in self.session.receive():
                    if not self.is_running: break # Check again inside loop
                    # if audio_data := response.data: # Assuming response.data is for audio
                    #     _LOGGER.info("Received audio response: (length %s) - ignored for now", len(audio_data))
                    if text := response.text:
                        _LOGGER.info(f"Received text: {text}")
                        # Here you would typically do something with the text,
                        # e.g., use TTS to speak it, display it, or send it as an event.
                    # Handle other modalities if configured
        except asyncio.CancelledError:
            _LOGGER.info("Receive task cancelled.")
        except Exception as e:
            if self.is_running: # Don't log as error if we are stopping
                 _LOGGER.error(f"Error in receive_task: {e}", exc_info=True)
        finally:
            _LOGGER.info("Receive task finished.")


    async def async_start(self):
        _LOGGER.info("async_start called")
        async with self._session_lock:
            if self.session:
                _LOGGER.info("Session already active. Not starting again.")
                return

            self.is_running = True # Ensure this is set
            self._stop_event.clear() # Clear stop flag
            self._muted_changed_event.clear() # Clear mute flag if set

            try:
                _LOGGER.info("Initializing Gemini Client and Session...")
                client = await self.hass.async_add_executor_job(
                    lambda: genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
                )
                self._session_manager = client.aio.live.connect(model=CONF_CHAT_MODEL, config=CONFIG_RESPONSE)
                self.session = await self._session_manager.__aenter__()
                _LOGGER.info("Gemini session started.")

                if self.stream_url:
                    _LOGGER.info("Opening AV stream...")
                    opened_successfully = await self.hass.async_add_executor_job(
                        self._open_av_stream_blocking, self.stream_url
                    )
                    if not opened_successfully:
                        _LOGGER.error("Failed to open AV stream. Cannot start audio capture.")
                        # Clean up session if stream opening fails catastrophically
                        await self.async_stop(called_from_start_failure=True)
                        return # Do not proceed to create tasks
                else:
                    _LOGGER.info("No stream_url configured.")


                # Create background tasks
                task_configs = []
                if self.stream_url and self.container: # Only if stream is ready
                     task_configs.append(("audio_capture_launcher", self._audio_capture_task_launcher()))
                task_configs.extend([
                    ("realtime_send", self._send_realtime_task()),
                    ("receive", self._receive_task()),
                ])

                for name, coro in task_configs:
                    _LOGGER.info(f"Creating background task: gemini_live_{name}")
                    task = self.hass.async_create_background_task(
                        coro,
                        f"gemini_live_{name}",
                        eager_start=True # Start immediately
                    )
                    self._tasks.add(task)

                _LOGGER.info(f"Gemini Live API tasks created: {len(self._tasks)}")

            except Exception as e:
                _LOGGER.error(f"Error starting Gemini Live: {e}", exc_info=True)
                await self.async_stop(called_from_start_failure=True) # Ensure cleanup
                # Do not re-raise, let Home Assistant handle it or manage state.


    async def async_stop(self, called_from_start_failure=False) -> None:
        _LOGGER.info(f"async_stop called. is_running: {self.is_running}, called_from_start_failure: {called_from_start_failure}")

        # If not called from start failure, it's an external stop, so set is_running to False.
        # If called from start failure, is_running might already be True, but we are aborting.
        # This flag primarily signals loops in tasks to terminate.
        if not called_from_start_failure:
            self.is_running = False
        self._stop_event.set() # Signal all tasks, especially blocking ones

        # Cancel the executor future for audio capture if it exists and is running
        if self._audio_capture_executor_future and not self._audio_capture_executor_future.done():
            _LOGGER.info("Cancelling audio capture executor future.")
            self._audio_capture_executor_future.cancel()
            # Give it a moment to acknowledge cancellation, but don't block HA shutdown too long
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
                # Wait for tasks to actually cancel
                # Add a timeout to prevent HA from hanging if a task doesn't cancel cleanly
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
                # Pass appropriate exception info if an error caused the stop
                await self._session_manager.__aexit__(None, None, None)
                _LOGGER.info("Gemini session closed.")
            except Exception as e_session:
                _LOGGER.error(f"Error closing Gemini session: {e_session}", exc_info=True)
            finally:
                self._session_manager = None
                self.session = None # Ensure session is None after manager aexit

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
        # self.is_running should already be False if stop was initiated externally.
        # If stop was due to start failure, ensure it's False.
        self.is_running = False