#!/usr/bin/env python3
"""Fake TRON2 robot WebSocket server for local integration tests.

This script emulates the small subset of the TRON2 robot WebSocket protocol
used by examples/tron2/robot_utils.py. It is intended for testing connection,
state polling, MoveJ, ServoJ, head, and gripper command flow when the real
robot is unavailable.

Example:
    uv run examples/tron2/fake_tron2_robot_server.py --host 0.0.0.0 --port 5000

Then start the VLA service against the fake robot:
    uv run examples/tron2/pi_client_service_static_finish.py --robot-ip 127.0.0.1
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed


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

ARM_JOINT_DIM = 14
SERVOJ_DIM = 16


@dataclass
class Args:
    host: str
    port: int
    log_level: str
    state_rate_hz: float


class FakeTron2Robot:
    """Stateful fake robot protocol endpoint."""

    def __init__(self, state_rate_hz: float) -> None:
        self.accid = "fake-tron2"
        self.joint_q = [*INIT_JOINTS_CLOTHES, *INIT_HEAD]
        self.left_opening = 100
        self.right_opening = 100
        self.mode = 0
        self.state_rate_hz = state_rate_hz
        self._clients: set[Any] = set()

    async def handle_connection(self, websocket: Any, _path: Any = None) -> None:
        peer = getattr(websocket, "remote_address", None)
        logging.info("Client connected: %s", peer)
        self._clients.add(websocket)

        notify_task = asyncio.create_task(self._periodic_notify(websocket))
        try:
            async for raw_message in websocket:
                await self._handle_raw_message(websocket, raw_message)
        except ConnectionClosed:
            logging.info("Client disconnected: %s", peer)
        finally:
            notify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await notify_task
            self._clients.discard(websocket)

    async def _periodic_notify(self, websocket: Any) -> None:
        """Send harmless robot-info notifications.

        robot_utils.py ignores notify_robot_info, but this makes the fake server
        feel closer to a live robot without affecting state queue semantics.
        """
        if self.state_rate_hz <= 0:
            return

        interval = 1.0 / self.state_rate_hz
        while True:
            await asyncio.sleep(interval)
            await self._send(
                websocket,
                "notify_robot_info",
                data={
                    "connected": True,
                    "mode": self.mode,
                    "fake": True,
                },
            )

    async def _handle_raw_message(self, websocket: Any, raw_message: Any) -> None:
        if not isinstance(raw_message, str):
            await self._send_error(websocket, None, "Only JSON text messages are supported.")
            return

        try:
            request = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            await self._send_error(websocket, None, f"Invalid JSON: {exc.msg}")
            return

        if not isinstance(request, dict):
            await self._send_error(websocket, None, "Request must be a JSON object.")
            return

        title = request.get("title")
        guid = request.get("guid")
        data = request.get("data") if isinstance(request.get("data"), dict) else {}

        if not isinstance(title, str):
            await self._send_error(websocket, guid, "Missing string field: title")
            return

        handler = {
            "request_get_joint_state": self._handle_get_joint_state,
            "request_get_limx_2fclaw_state": self._handle_get_gripper_state,
            "request_get_move_pose": self._handle_get_move_pose,
            "request_set_servo_mode": self._handle_set_servo_mode,
            "request_movej": self._handle_movej,
            "request_servoj": self._handle_servoj,
            "request_moveh": self._handle_moveh,
            "request_set_limx_2fclaw_cmd": self._handle_set_gripper,
            "request_movep": self._handle_movep,
            "request_servop": self._handle_servop,
            "request_set_servop_mode": self._handle_set_servop_mode,
            "request_emgy_stop": self._handle_emergency_stop,
            "request_light_effect": self._handle_light_effect,
        }.get(title)

        if handler is None:
            logging.warning("Unsupported request title: %s", title)
            await self._send_error(websocket, guid, f"Unsupported request title: {title}")
            return

        await handler(websocket, guid, data)

    async def _handle_get_joint_state(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        del data
        await self._send(websocket, "response_get_joint_state", guid=guid, data={"q": list(self.joint_q)})

    async def _handle_get_gripper_state(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        del data
        await self._send(
            websocket,
            "response_get_limx_2fclaw_state",
            guid=guid,
            data={
                "left_opening": self.left_opening,
                "right_opening": self.right_opening,
            },
        )

    async def _handle_get_move_pose(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        del data
        await self._send(
            websocket,
            "response_get_move_pose",
            guid=guid,
            data={
                "left_position": [0.35, 0.20, 0.30],
                "left_quat": [1.0, 0.0, 0.0, 0.0],
                "right_position": [0.35, -0.20, 0.30],
                "right_quat": [1.0, 0.0, 0.0, 0.0],
            },
        )

    async def _handle_set_servo_mode(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        self.mode = int(data.get("mode", self.mode))
        logging.info("Set servo mode: %s", self.mode)
        await self._send_ack(websocket, "response_set_servo_mode", guid)

    async def _handle_set_servop_mode(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        del data
        self.mode = 2
        logging.info("Set ServoP mode")
        await self._send_ack(websocket, "response_set_servop_mode", guid)

    async def _handle_movej(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        joints = data.get("joint", [])
        if not self._is_number_list(joints, ARM_JOINT_DIM):
            await self._send_error(websocket, guid, "request_movej expects data.joint with 14 numbers.")
            return

        self.joint_q[:ARM_JOINT_DIM] = [float(value) for value in joints]
        logging.info("MoveJ accepted: time=%s", data.get("time"))
        await self._send_ack(websocket, "response_movej", guid)

    async def _handle_servoj(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        joints = data.get("q", [])
        if not self._is_number_list(joints, SERVOJ_DIM):
            await self._send_error(websocket, guid, "request_servoj expects data.q with 16 numbers.")
            return

        self.joint_q = [float(value) for value in joints]
        logging.debug("ServoJ accepted")
        await self._send_ack(websocket, "response_servoj", guid)

    async def _handle_moveh(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        head = data.get("joint", [])
        if not self._is_number_list(head, 2):
            await self._send_error(websocket, guid, "request_moveh expects data.joint with 2 numbers.")
            return

        self.joint_q[14:16] = [float(value) for value in head]
        logging.info("MoveH accepted: time=%s", data.get("time"))
        await self._send_ack(websocket, "response_moveh", guid)

    async def _handle_set_gripper(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        self.left_opening = self._clip_opening(data.get("left_opening", self.left_opening))
        self.right_opening = self._clip_opening(data.get("right_opening", self.right_opening))
        logging.info("Set gripper: left=%s right=%s", self.left_opening, self.right_opening)
        await self._send_ack(websocket, "response_set_limx_2fclaw_cmd", guid)

    async def _handle_movep(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        logging.info("MoveP accepted: time=%s", data.get("time"))
        await self._send_ack(websocket, "response_movep", guid)

    async def _handle_servop(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        logging.debug("ServoP accepted: time=%s", data.get("time"))
        await self._send_ack(websocket, "response_servop", guid)

    async def _handle_emergency_stop(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        del data
        logging.warning("Emergency stop requested")
        await self._send_ack(websocket, "response_emgy_stop", guid)

    async def _handle_light_effect(self, websocket: Any, guid: str | None, data: dict[str, Any]) -> None:
        logging.info("Light effect requested: %s", data.get("effect"))
        await self._send_ack(websocket, "response_light_effect", guid)

    async def _send_ack(self, websocket: Any, title: str, guid: str | None) -> None:
        await self._send(websocket, title, guid=guid, data={"success": True})

    async def _send_error(self, websocket: Any, guid: str | None, error: str) -> None:
        await self._send(
            websocket,
            "response_error",
            guid=guid,
            data={
                "success": False,
                "error": error,
            },
        )

    async def _send(
        self,
        websocket: Any,
        title: str,
        *,
        guid: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        response = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data or {},
        }
        await websocket.send(json.dumps(response))

    @staticmethod
    def _is_number_list(value: Any, length: int) -> bool:
        if not isinstance(value, list) or len(value) != length:
            return False
        return all(isinstance(item, (int, float)) for item in value)

    @staticmethod
    def _clip_opening(value: Any) -> int:
        if not isinstance(value, (int, float)):
            return 100
        return int(max(0, min(100, value)))


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Fake TRON2 robot WebSocket server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--state-rate-hz",
        type=float,
        default=50.0,
        help="Rate for harmless notify_robot_info messages; request/response state polling is immediate.",
    )
    args = parser.parse_args()

    if args.port <= 0:
        parser.error("--port must be > 0")
    if args.state_rate_hz < 0:
        parser.error("--state-rate-hz must be >= 0")

    return Args(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        state_rate_hz=args.state_rate_hz,
    )


async def main_async(args: Args) -> None:
    robot = FakeTron2Robot(state_rate_hz=args.state_rate_hz)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        logging.info("Shutdown requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_shutdown)

    logging.info("Fake TRON2 robot listening on ws://%s:%s", args.host, args.port)
    async with websockets.serve(
        robot.handle_connection,
        args.host,
        args.port,
        ping_interval=20,
        ping_timeout=20,
        max_size=None,
    ):
        await stop_event.wait()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
