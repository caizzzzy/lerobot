import csv
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .configs import AsyncRecorderConfig


class AsyncInferenceRecorder:
    """Best-effort asynchronous recorder for client-side experiments.

    The control loop only enqueues lightweight events and never blocks on disk I/O.
    When the queue is full, the oldest pending event is dropped to protect inference latency.
    """

    def __init__(
        self,
        config: AsyncRecorderConfig,
        client_fps: int,
        logger: logging.Logger,
        frame_provider: Callable[[], dict[str, np.ndarray]] | None = None,
    ):
        self.config = config
        self.client_fps = client_fps
        self.logger = logger
        self.frame_provider = frame_provider

        self.event_queue: Queue[dict[str, Any] | None] = Queue(maxsize=config.queue_size)
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.capture_thread: threading.Thread | None = None

        self.run_dir: Path | None = None
        self.video_dir: Path | None = None
        self.actions_path: Path | None = None
        self.observations_path: Path | None = None

        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.video_writer_started_at: dict[str, float] = {}
        self.video_writer_segment_idx: dict[str, int] = {}
        self.action_file = None
        self.action_writer: csv.DictWriter | None = None
        self.observation_file = None
        self.observation_writer: csv.DictWriter | None = None

        self.dropped_events = 0
        self.recorded_frames = 0
        self.recorded_actions = 0
        self.recorded_observations = 0

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def start(self) -> None:
        if not self.enabled or self.worker_thread is not None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.output_dir).expanduser() / f"run_{timestamp}"
        self.video_dir = self.run_dir / "videos"
        self.actions_path = self.run_dir / "actions.csv"
        self.observations_path = self.run_dir / "observations.csv"

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata()

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, name="async-inference-recorder", daemon=True)
        self.worker_thread.start()
        if self.frame_provider is not None:
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                name="async-inference-video-capture",
                daemon=True,
            )
            self.capture_thread.start()
        self.logger.info(f"Async recorder enabled. Saving records to {self.run_dir}")

    def stop(self) -> None:
        if not self.enabled or self.worker_thread is None:
            return

        self.stop_event.set()
        self._enqueue_internal(None)
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        self.worker_thread.join()
        self.worker_thread = None

        self._close_resources()
        self._write_summary()
        self.logger.info(
            "Async recorder stopped. "
            f"observations={self.recorded_observations}, actions={self.recorded_actions}, dropped={self.dropped_events}"
        )

    def log_observation(
        self,
        observation: dict[str, Any],
        timestamp: float,
        timestep: int,
        queue_size: int,
        must_go: bool,
    ) -> None:
        if not self.enabled:
            return

        frames = {}
        for camera_key in self.config.camera_keys:
            frame = observation.get(camera_key)
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                frames[camera_key] = frame

        event = {
            "type": "observation",
            "timestamp": timestamp,
            "timestep": timestep,
            "queue_size": queue_size,
            "must_go": must_go,
        }
        if self.frame_provider is None:
            event["frames"] = frames
        self._enqueue_event(event)

    def log_action(
        self,
        action_timestep: int,
        action_source_timestamp: float,
        action_execute_timestamp: float,
        requested_action: dict[str, float],
        applied_action: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        event = {
            "type": "action",
            "timestep": action_timestep,
            "source_timestamp": action_source_timestamp,
            "execute_timestamp": action_execute_timestamp,
            "requested_action": requested_action,
            "applied_action": {
                key: float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
                for key, value in applied_action.items()
            },
        }
        self._enqueue_event(event)

    def _enqueue_event(self, event: dict[str, Any]) -> None:
        try:
            self.event_queue.put_nowait(event)
        except Full:
            try:
                self.event_queue.get_nowait()
            except Empty:
                pass
            self.dropped_events += 1
            self._enqueue_internal(event)

    def _enqueue_internal(self, event: dict[str, Any] | None) -> None:
        while True:
            try:
                self.event_queue.put_nowait(event)
                return
            except Full:
                try:
                    self.event_queue.get_nowait()
                except Empty:
                    time.sleep(0.001)

    def _worker_loop(self) -> None:
        while True:
            try:
                event = self.event_queue.get(timeout=0.2)
            except Empty:
                if self.stop_event.is_set():
                    break
                continue

            if event is None:
                if self.stop_event.is_set() and self.event_queue.empty():
                    break
                continue

            try:
                if event["type"] == "observation":
                    self._handle_observation(event)
                elif event["type"] == "action":
                    self._handle_action(event)
                elif event["type"] == "video_frame":
                    self._handle_video_frame(event)
            except Exception as exc:  # nosec B110
                self.logger.warning(f"Recorder worker failed to persist event: {exc}")

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / float(self.config.video_fps or self.client_fps)
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            frames = {}
            try:
                provided_frames = self.frame_provider() if self.frame_provider is not None else {}
                for camera_key in self.config.camera_keys:
                    frame = provided_frames.get(camera_key)
                    if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        frames[camera_key] = frame.copy()
            except Exception as exc:  # nosec B110
                self.logger.debug(f"Frame provider failed during recording: {exc}")

            if frames:
                self._enqueue_event(
                    {
                        "type": "video_frame",
                        "timestamp": time.time(),
                        "frames": frames,
                    }
                )

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, frame_interval - elapsed))

    def _handle_observation(self, event: dict[str, Any]) -> None:
        self._ensure_observation_writer()

        row = {
            "timestamp": event["timestamp"],
            "timestep": event["timestep"],
            "queue_size": event["queue_size"],
            "must_go": int(bool(event["must_go"])),
        }
        self.observation_writer.writerow(row)
        self.observation_file.flush()
        self.recorded_observations += 1

        for camera_key, frame in event.get("frames", {}).items():
            self._write_video_frame(camera_key, frame)

    def _handle_action(self, event: dict[str, Any]) -> None:
        self._ensure_action_writer(event["requested_action"], event["applied_action"])

        row = {
            "timestep": event["timestep"],
            "source_timestamp": event["source_timestamp"],
            "execute_timestamp": event["execute_timestamp"],
        }
        for key, value in event["requested_action"].items():
            row[f"requested.{key}"] = value
        for key, value in event["applied_action"].items():
            row[f"applied.{key}"] = value

        self.action_writer.writerow(row)
        self.action_file.flush()
        self.recorded_actions += 1

    def _handle_video_frame(self, event: dict[str, Any]) -> None:
        for camera_key, frame in event["frames"].items():
            self._write_video_frame(camera_key, frame)

    def _write_video_frame(self, camera_key: str, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer = self._get_or_create_video_writer(camera_key, frame)
        writer.write(frame)
        self.recorded_frames += 1

    def _ensure_action_writer(self, requested_action: dict[str, float], applied_action: dict[str, Any]) -> None:
        if self.action_writer is not None:
            return

        fieldnames = ["timestep", "source_timestamp", "execute_timestamp"]
        fieldnames.extend(f"requested.{key}" for key in requested_action.keys())
        fieldnames.extend(f"applied.{key}" for key in applied_action.keys())

        self.action_file = self.actions_path.open("w", newline="", encoding="utf-8")
        self.action_writer = csv.DictWriter(self.action_file, fieldnames=fieldnames)
        self.action_writer.writeheader()
        self.action_file.flush()

    def _ensure_observation_writer(self) -> None:
        if self.observation_writer is not None:
            return

        self.observation_file = self.observations_path.open("w", newline="", encoding="utf-8")
        self.observation_writer = csv.DictWriter(
            self.observation_file,
            fieldnames=["timestamp", "timestep", "queue_size", "must_go"],
        )
        self.observation_writer.writeheader()
        self.observation_file.flush()

    def _get_or_create_video_writer(self, camera_key: str, frame: np.ndarray) -> cv2.VideoWriter:
        writer = self.video_writers.get(camera_key)
        now = time.time()
        if writer is not None:
            started_at = self.video_writer_started_at.get(camera_key, now)
            if now - started_at < self.config.segment_seconds:
                return writer
            writer.release()
            self.video_writers.pop(camera_key, None)
            self.video_writer_started_at.pop(camera_key, None)

        segment_idx = self.video_writer_segment_idx.get(camera_key, 0)
        self.video_writer_segment_idx[camera_key] = segment_idx + 1

        if self.config.segment_seconds > 0:
            video_name = f"{camera_key}_{segment_idx:05d}.{self.config.video_extension.lstrip('.')}"
        else:
            video_name = f"{camera_key}.{self.config.video_extension.lstrip('.')}"

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        video_path = self.video_dir / video_name
        fps = float(self.config.video_fps or self.client_fps)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        self.video_writers[camera_key] = writer
        self.video_writer_started_at[camera_key] = now
        return writer

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "camera_keys": self.config.camera_keys,
            "queue_size": self.config.queue_size,
            "video_fps": self.config.video_fps or self.client_fps,
            "video_codec": self.config.video_codec,
        }
        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_summary(self) -> None:
        if self.run_dir is None:
            return

        summary = {
            "recorded_observations": self.recorded_observations,
            "recorded_actions": self.recorded_actions,
            "recorded_frames": self.recorded_frames,
            "dropped_events": self.dropped_events,
        }
        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _close_resources(self) -> None:
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        self.video_writer_started_at.clear()

        if self.action_file is not None:
            self.action_file.close()
            self.action_file = None
            self.action_writer = None

        if self.observation_file is not None:
            self.observation_file.close()
            self.observation_file = None
            self.observation_writer = None
