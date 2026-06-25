#!/usr/bin/env python3
"""
TRON2 VLA WebSocket service (Orin side).

Architecture
------------
- This process is a WebSocket SERVER on the Orin, default ws://0.0.0.0:8765.
- The chassis/controller connects here and sends:
    {"cmd": "run",    "task": "grasp_from_table"}
    {"cmd": "stop"}
    {"cmd": "status"}
- Two independent VLA policy servers are already running:
    model 1: grasp_from_table / place_to_shelf
    model 2: grasp_from_shelf / place_to_table

The robot environment is initialized exactly once and shared by both models.
Do NOT start a second Tron2Env for model 2: that would create conflicting
robot-control connections.

Examples
--------
# Both VLA servers run locally on the Orin.
python examples/tron2/pi_ws_service.py \
  --robot-ip 10.192.1.2 \
  --model1-host 127.0.0.1 --model1-port 8000 \
  --model2-host 127.0.0.1 --model2-port 8001 \
  --service-host 0.0.0.0 --service-port 8765

# Test from another machine:
#   websocat ws://<ORIN_IP>:8765
#   {"cmd":"status"}
#   {"cmd":"run","task":"grasp_from_table"}
#   {"cmd":"stop"}
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import einops
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from openpi_client import image_tools, websocket_client_policy
from real_env import EnvConfig, Tron2Env
from robot_utils import Tron2Config


# ---------------------------------------------------------------------------
# Task / model routing
# ---------------------------------------------------------------------------
# Model 1 performs delivery: table -> shelf.
# Model 2 performs recovery: shelf -> table.
TASK_TO_MODEL = {
    "grasp_from_table": "model1",
    "place_to_shelf": "model1",
    "grasp_from_shelf": "model2",
    "place_to_table": "model2",
}

TASK_PROMPTS = {
    "grasp_from_table": "grasp the object from the table",
    "place_to_shelf": "place the object into the shelf box",
    "grasp_from_shelf": "grasp the object from the shelf box",
    "place_to_table": "place the object back to the table box",
}

VALID_TASKS = frozenset(TASK_TO_MODEL)

# Keep these initial poses identical to the original pi_client.py.
INIT_JOINTS_CLOTHES = [
    0.026899,
    0.2612,
    -0.02709991,
    -1.5477003,
    0.265,
    0.0180999,
    -0.0614999,
    0.008999,
    -0.269,
    0.02069998,
    -1.5567001,
    -0.254,
    -0.02309972,
    0.06469989,
]
INIT_HEAD = [1.0467, -0.0139998]


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime configuration for the robot and both policy servers."""

    robot_ip: str
    model1_host: str
    model1_port: int
    model2_host: str
    model2_port: int
    service_host: str
    service_port: int
    max_infer_rounds: int
    status_every_actions: int
    action_interval_s: float
    reset_on_stop: bool
    static_finish_enabled: bool
    static_action_threshold: float
    static_action_frames_required: int
    min_actions_before_static_finish: int
    inject_task_prompt: bool
    prompt_key: str


class VLAWebSocketService:
    """Owns one TRON2 environment and routes each task to the correct VLA model."""

    def __init__(self, env: Tron2Env, config: ServiceConfig) -> None:
        self.env = env
        self.config = config

        # The policy clients are created once and reused; model selection happens
        # per task according to TASK_TO_MODEL.
        self.policies = {
            "model1": websocket_client_policy.WebsocketClientPolicy(
                host=config.model1_host,
                port=config.model1_port,
            ),
            "model2": websocket_client_policy.WebsocketClientPolicy(
                host=config.model2_host,
                port=config.model2_port,
            ),
        }

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: set[Any] = set()
        self._shutdown_event = asyncio.Event()

        # Robot inference is synchronous/blocking, so it runs in a worker thread.
        # Commands are still handled immediately by the asyncio WebSocket loop.
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._running = False
        self._active_task: Optional[str] = None
        self._step = 0
        self._last_message = "idle"

    # -----------------------------------------------------------------------
    # WebSocket transport
    # -----------------------------------------------------------------------
    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        logging.info(
            "VLA service listening on ws://%s:%s", 
            self.config.service_host,
            self.config.service_port,
        )

        # max_size=None avoids rejecting messages merely because a client sends
        # an unusually large status/debug payload. Incoming command messages are
        # still validated below.
        async with websockets.serve(
            self._handle_connection,
            self.config.service_host,
            self.config.service_port,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ):
            await self._shutdown_event.wait()

        await self._wait_for_running_task()

    async def shutdown(self) -> None:
        """Request a safe stop, then close the server once the worker exits."""
        await self._request_stop()
        self._shutdown_event.set()

    async def _handle_connection(self, websocket: Any, _path: Any = None) -> None:
        """Accept a chassis/controller connection and handle JSON commands."""
        peer = getattr(websocket, "remote_address", None)
        logging.info("WebSocket client connected: %s", peer)
        self._clients.add(websocket)

        try:
            await self._send_to(websocket, self._status_payload())
            async for raw_message in websocket:
                await self._handle_raw_command(websocket, raw_message)
        except ConnectionClosed:
            logging.info("WebSocket client disconnected: %s", peer)
        finally:
            self._clients.discard(websocket)

    async def _handle_raw_command(self, websocket: Any, raw_message: Any) -> None:
        if not isinstance(raw_message, str):
            await self._send_to(
                websocket,
                self._error_payload(None, "Only UTF-8 JSON text messages are supported."),
            )
            return

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            await self._send_to(
                websocket,
                self._error_payload(None, f"Invalid JSON: {exc.msg}"),
            )
            return

        if not isinstance(payload, dict):
            await self._send_to(
                websocket,
                self._error_payload(None, "Command payload must be a JSON object."),
            )
            return

        cmd = payload.get("cmd")
        if cmd == "run":
            task = payload.get("task")
            await self._start_task(task)
            return

        if cmd == "stop":
            await self._request_stop()
            return

        if cmd == "status":
            await self._send_to(websocket, self._status_payload())
            return

        await self._send_to(
            websocket,
            self._error_payload(
                payload.get("task"),
                "Unsupported command. Expected one of: run, stop, status.",
            ),
        )

    async def _send_to(self, websocket: Any, payload: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed:
            self._clients.discard(websocket)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        """Send task responses to every currently connected controller."""
        message = json.dumps(payload, ensure_ascii=False)
        disconnected: list[Any] = []

        for websocket in list(self._clients):
            try:
                await websocket.send(message)
            except ConnectionClosed:
                disconnected.append(websocket)

        for websocket in disconnected:
            self._clients.discard(websocket)

    def _emit_from_worker(self, payload: dict[str, Any]) -> None:
        """Thread-safe bridge from blocking robot inference to async WebSocket I/O."""
        if self._loop is None or self._loop.is_closed():
            return

        def schedule_send() -> None:
            asyncio.create_task(self._broadcast(payload))

        self._loop.call_soon_threadsafe(schedule_send)

    # -----------------------------------------------------------------------
    # Command / state handling
    # -----------------------------------------------------------------------
    async def _start_task(self, task: Any) -> None:
        if not isinstance(task, str) or task not in VALID_TASKS:
            await self._broadcast(
                self._error_payload(
                    task if isinstance(task, str) else None,
                    "Unknown task. Allowed tasks: " + ", ".join(sorted(VALID_TASKS)),
                )
            )
            return

        with self._state_lock:
            if self._running:
                busy_task = self._active_task
            else:
                busy_task = None
                self._running = True
                self._active_task = task
                self._step = 0
                self._last_message = "model starting"
                self._stop_event.clear()

        if busy_task is not None:
            await self._broadcast(
                self._error_payload(
                    task,
                    f"Cannot start {task}: task {busy_task} is still running.",
                )
            )
            return

        model_name = TASK_TO_MODEL[task]
        logging.info(
            "Starting task=%s with %s (%s)",
            task,
            model_name,
            TASK_PROMPTS[task],
        )
        await self._broadcast(self._status_payload())

        # Do not await this here. The receive loop must remain responsive to
        # cmd=stop and cmd=status while model inference is running.
        self._worker_task = asyncio.create_task(
            asyncio.to_thread(self._execute_task_sync, task),
            name=f"vla-{task}",
        )

    async def _request_stop(self) -> None:
        with self._state_lock:
            running = self._running
            task = self._active_task
            if running:
                self._last_message = "stop requested"
                self._stop_event.set()

        if not running:
            # The protocol still gets a deterministic reply if it sends stop
            # while idle.
            await self._broadcast(
                {
                    "type": "stopped",
                    "task": None,
                    "running": False,
                    "success": False,
                    "message": "当前没有运行中的任务",
                }
            )
            return

        # The worker will emit the final stopped message only after it has left
        # the current safe execution boundary and optionally reset the robot.
        await self._broadcast(self._status_payload(task_override=task))

    async def _wait_for_running_task(self) -> None:
        worker = self._worker_task
        if worker is None:
            return

        if not worker.done():
            with self._state_lock:
                self._stop_event.set()
                self._last_message = "service shutting down"
            with contextlib.suppress(Exception):
                await worker

    def _status_payload(self, task_override: Optional[str] = None) -> dict[str, Any]:
        with self._state_lock:
            task = task_override if task_override is not None else self._active_task
            return {
                "type": "status",
                "task": task,
                "running": self._running,
                "step": self._step,
                "message": self._last_message,
            }

    @staticmethod
    def _error_payload(task: Optional[str], error: str) -> dict[str, Any]:
        return {
            "type": "error",
            "task": task,
            "running": False,
            "success": False,
            "error": error,
            "message": "任务执行失败",
        }

    # -----------------------------------------------------------------------
    # Synchronous robot/model execution (worker thread)
    # -----------------------------------------------------------------------
    def _execute_task_sync(self, task: str) -> None:
        """Run one task until EOS, near-static actions, stop, timeout, or error.

        Near-static completion is evaluated on the *executed action stream*:
        after enough consecutive action vectors have maximum element-wise change
        below ``static_action_threshold``, the task is treated as complete and
        a protocol ``done`` response is returned.  This is deliberately based
        on the policy output, not on robot state feedback.
        """
        previous_action: Optional[np.ndarray] = None
        static_action_frames = 0

        try:
            policy = self.policies[TASK_TO_MODEL[task]]

            for infer_round in range(1, self.config.max_infer_rounds + 1):
                if self._stop_event.is_set():
                    self._finish_stopped(task)
                    return

                obs = self.env.get_obs()
                obs = self._preprocess_observation(obs, task)

                infer_start = time.monotonic()
                answer = policy.infer(obs)
                infer_duration = time.monotonic() - infer_start

                actions = self._extract_actions(answer)
                logging.info(
                    "task=%s round=%d model=%s actions=%d infer=%.3fs",
                    task,
                    infer_round,
                    TASK_TO_MODEL[task],
                    len(actions),
                    infer_duration,
                )

                for action in actions:
                    if self._stop_event.is_set():
                        self._finish_stopped(task)
                        return

                    # Preserve the existing control behavior: execute the
                    # current target first, then decide whether the policy has
                    # become effectively stationary.
                    self.env.step(action)

                    action_vector = np.asarray(action, dtype=np.float64).reshape(-1)
                    max_action_delta: Optional[float] = None
                    if (
                        self.config.static_finish_enabled
                        and previous_action is not None
                        and action_vector.shape == previous_action.shape
                    ):
                        max_action_delta = float(
                            np.max(np.abs(action_vector - previous_action))
                        )
                        if max_action_delta <= self.config.static_action_threshold:
                            static_action_frames += 1
                        else:
                            static_action_frames = 0
                    elif previous_action is not None:
                        # A policy server should not change action dimensionality
                        # mid-task.  Do not silently carry a stale static count.
                        static_action_frames = 0

                    previous_action = action_vector.copy()

                    with self._state_lock:
                        self._step += 1
                        current_step = self._step
                        self._last_message = "model running"

                    if current_step % self.config.status_every_actions == 0:
                        self._emit_from_worker(self._status_payload())

                    if (
                        self.config.static_finish_enabled
                        and current_step >= self.config.min_actions_before_static_finish
                        and static_action_frames >= self.config.static_action_frames_required
                    ):
                        logging.info(
                            "task=%s finished by near-static actions: step=%d, "
                            "consecutive_frames=%d, max_delta=%s, threshold=%.6f",
                            task,
                            current_step,
                            static_action_frames,
                            "n/a" if max_action_delta is None else f"{max_action_delta:.6f}",
                            self.config.static_action_threshold,
                        )
                        self._finish_done(task, "检测到动作接近静止，任务完成")
                        return

                    if self.config.action_interval_s > 0:
                        time.sleep(self.config.action_interval_s)

                # Prefer an explicit completion signal when the policy server
                # provides one. This is more reliable than any heuristic.
                if self._policy_answer_is_done(answer):
                    self._finish_done(task)
                    return

            # No EOS/done field is present in the original pi_client.py. Therefore
            # max_infer_rounds is the deterministic fallback completion boundary.
            self._finish_done(task)

        except Exception as exc:  # noqa: BLE001 - needs to report remote failures
            logging.exception("Task failed: %s", task)
            if self._stop_event.is_set():
                self._finish_stopped(task)
                return

            with self._state_lock:
                self._running = False
                self._active_task = None
                self._last_message = "task failed"
            self._emit_from_worker(self._error_payload(task, repr(exc)))

    def _preprocess_observation(self, obs: dict[str, Any], task: str) -> dict[str, Any]:
        """Apply image conversion and attach the task prompt when configured."""
        if "images" not in obs or not isinstance(obs["images"], dict):
            raise KeyError("Observation must contain obs['images'] as a dict.")

        if self.config.inject_task_prompt:
            # OpenPI-style policy servers commonly consume a `prompt` field.
            # Set --prompt-key to match your server's exact observation schema,
            # or use --no-inject-task-prompt if the server binds a prompt itself.
            obs[self.config.prompt_key] = TASK_PROMPTS[task]

        for camera_name, image in obs["images"].items():
            image_uint8 = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224)
            )
            obs["images"][camera_name] = einops.rearrange(
                image_uint8,
                "h w c -> c h w",
            )

        return obs


    @staticmethod
    def _policy_answer_is_done(answer: Any) -> bool:
        """Read common optional completion fields from a policy response.

        The base pi_client.py only uses `actions`, so this remains backward
        compatible if the server does not expose a completion field.
        """
        if not isinstance(answer, dict):
            return False
        return any(bool(answer.get(key, False)) for key in ("done", "is_done", "completed"))

    @staticmethod
    def _extract_actions(answer: Any) -> np.ndarray:
        if not isinstance(answer, dict) or "actions" not in answer:
            raise KeyError("Policy response must be a dict containing the key 'actions'.")

        actions = np.asarray(answer["actions"])
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"Invalid action plan shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned NaN or Inf action values.")

        return actions

    def _finish_done(self, task: str, message: str = "搬运完成") -> None:
        with self._state_lock:
            self._running = False
            self._active_task = None
            self._last_message = "task completed"

        self._emit_from_worker(
            {
                "type": "done",
                "task": task,
                "running": False,
                "success": True,
                "message": message,
            }
        )

    def _finish_stopped(self, task: str) -> None:
        if self.config.reset_on_stop:
            try:
                # In the original script, env.reset() moves the system to its
                # configured initial pose. Keep this behind a flag because exact
                # reset semantics depend on your Tron2Env implementation.
                logging.info("Stop requested: returning robot to configured reset pose.")
                self.env.reset()
            except Exception:  # noqa: BLE001 - stop response must still be sent
                logging.exception("env.reset() failed while stopping task=%s", task)

        with self._state_lock:
            self._running = False
            self._active_task = None
            self._last_message = "task stopped"
            self._stop_event.clear()

        self._emit_from_worker(
            {
                "type": "stopped",
                "task": task,
                "running": False,
                "success": False,
                "message": "任务已停止",
            }
        )


# ---------------------------------------------------------------------------
# Process setup
# ---------------------------------------------------------------------------
def parse_args() -> ServiceConfig:
    parser = argparse.ArgumentParser(
        description="TRON2 dual-VLA WebSocket service on the Orin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--robot-ip", default="10.192.1.2")

    # VLA server 1: table -> shelf tasks.
    parser.add_argument("--model1-host", default="127.0.0.1")
    parser.add_argument("--model1-port", type=int, default=8000)

    # VLA server 2: shelf -> table tasks.
    parser.add_argument("--model2-host", default="127.0.0.1")
    parser.add_argument("--model2-port", type=int, default=8001)

    # Orin's externally callable service endpoint.
    parser.add_argument("--service-host", default="0.0.0.0")
    parser.add_argument("--service-port", type=int, default=8765)

    # A policy inference round returns an action chunk. In the original script
    # the outer loop never increments t, so this must be made explicit.
    parser.add_argument("--max-infer-rounds", type=int, default=100)
    parser.add_argument("--status-every-actions", type=int, default=1)
    parser.add_argument("--action-interval-s", type=float, default=0.03)

    # Stop safety. Set --no-reset-on-stop only after you have validated another
    # hardware-safe posture method.
    parser.add_argument(
        "--no-reset-on-stop",
        action="store_true",
        help="Do not call env.reset() after cmd=stop.",
    )

    # Near-static action completion.  The policy is considered finished after
    # N consecutive *executed* action vectors differ from the previous action
    # by no more than the configured max element-wise threshold.
    parser.add_argument(
        "--disable-static-finish",
        action="store_true",
        help="Disable near-static action detection and rely on EOS/max rounds only.",
    )
    parser.add_argument(
        "--static-action-threshold",
        type=float,
        default=0.002,
        help="Max absolute per-dimension action delta treated as stationary.",
    )
    parser.add_argument(
        "--static-action-frames-required",
        type=int,
        default=10,
        help="Consecutive near-static action deltas required before sending done.",
    )
    parser.add_argument(
        "--min-actions-before-static-finish",
        type=int,
        default=30,
        help="Do not apply near-static completion before this many actions execute.",
    )
    parser.add_argument(
        "--no-inject-task-prompt",
        action="store_true",
        help="Do not add the per-task language prompt to the policy observation.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Observation field name used by the VLA policy server for language input.",
    )

    args = parser.parse_args()

    if args.max_infer_rounds <= 0:
        parser.error("--max-infer-rounds must be > 0")
    if args.status_every_actions <= 0:
        parser.error("--status-every-actions must be > 0")
    if args.action_interval_s < 0:
        parser.error("--action-interval-s must be >= 0")
    if args.static_action_threshold <= 0:
        parser.error("--static-action-threshold must be > 0")
    if args.static_action_frames_required <= 0:
        parser.error("--static-action-frames-required must be > 0")
    if args.min_actions_before_static_finish < 0:
        parser.error("--min-actions-before-static-finish must be >= 0")

    return ServiceConfig(
        robot_ip=args.robot_ip,
        model1_host=args.model1_host,
        model1_port=args.model1_port,
        model2_host=args.model2_host,
        model2_port=args.model2_port,
        service_host=args.service_host,
        service_port=args.service_port,
        max_infer_rounds=args.max_infer_rounds,
        status_every_actions=args.status_every_actions,
        action_interval_s=args.action_interval_s,
        reset_on_stop=not args.no_reset_on_stop,
        static_finish_enabled=not args.disable_static_finish,
        static_action_threshold=args.static_action_threshold,
        static_action_frames_required=args.static_action_frames_required,
        min_actions_before_static_finish=args.min_actions_before_static_finish,
        inject_task_prompt=not args.no_inject_task_prompt,
        prompt_key=args.prompt_key,
    )


def build_environment(config: ServiceConfig) -> Tron2Env:
    robot_config = Tron2Config(
        robot_ip=config.robot_ip,
        init_joints=INIT_JOINTS_CLOTHES,
        init_head=INIT_HEAD,
    )
    env_config = EnvConfig(
        robot_config=robot_config,
        interp_points=8,
        time_sync_tolerance=0.01,
    )
    return Tron2Env(env_config)


async def run_service(config: ServiceConfig) -> None:
    env = build_environment(config)
    with env:
        logging.info("Initializing TRON2 environment.")
        env.reset()

        service = VLAWebSocketService(env, config)
        loop = asyncio.get_running_loop()

        def request_shutdown() -> None:
            logging.info("Shutdown signal received.")
            asyncio.create_task(service.shutdown())

        # add_signal_handler is not available on Windows' Proactor loop. The
        # script is intended for Orin/Linux, but this keeps local syntax tests
        # and limited Windows use workable.
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, request_shutdown)

        await service.serve()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = parse_args()
    asyncio.run(run_service(config))


if __name__ == "__main__":
    main()
