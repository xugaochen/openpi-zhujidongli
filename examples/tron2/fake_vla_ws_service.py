#!/usr/bin/env python3
"""
Fake VLA WebSocket service.

不连接真实模型，不连接机器人。
用于联调底盘/上位机对 VLA 服务的 run / stop / status 调用。

安装依赖：
    pip install websockets

启动：
    python fake_vla_ws_service.py --host 0.0.0.0 --port 8765

支持命令：
    {"cmd": "run", "task": "grasp_from_table"}
    {"cmd": "stop"}
    {"cmd": "status"}
"""

import argparse
import asyncio
import json
import logging
import signal
from dataclasses import dataclass, field
from typing import Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

VALID_TASKS = {
    "grasp_from_table",
    "place_to_shelf",
    "grasp_from_shelf",
    "place_to_table",
}


@dataclass
class ServiceState:
    running: bool = False
    task: Optional[str] = None
    step: int = 0
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    task_runner: Optional[asyncio.Task] = None
    clients: Set[object] = field(default_factory=set)


class FakeVLAService:
    def __init__(self, steps: int, step_interval: float, status_every_steps: int):
        self.steps = steps
        self.step_interval = step_interval
        self.status_every_steps = status_every_steps
        self.state = ServiceState()
        self.lock = asyncio.Lock()

    def status_payload(self, message: str = "fake model running") -> dict:
        return {
            "type": "status",
            "task": self.state.task,
            "running": self.state.running,
            "step": self.state.step,
            "message": message,
        }

    async def broadcast(self, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        dead = []
        for client in list(self.state.clients):
            try:
                await client.send(text)
            except ConnectionClosed:
                dead.append(client)
        for client in dead:
            self.state.clients.discard(client)

    async def run_task(self, task: str) -> None:
        try:
            print(f"[Fake VLA] 调用模型：task={task}")
            print(f"[Fake VLA] 开始运行，共 {self.steps} step")

            for step in range(1, self.steps + 1):
                if self.state.stop_event.is_set():
                    print("[Fake VLA] 收到 stop，停止执行")
                    self.state.running = False
                    await self.broadcast({
                        "type": "stopped",
                        "task": task,
                        "running": False,
                        "success": False,
                        "message": "假模型任务已停止",
                    })
                    return

                self.state.step = step
                print(f"[Fake VLA] 运行 step {step}/{self.steps}")
                if step == 1 or step % self.status_every_steps == 0:
                    await self.broadcast(self.status_payload())

                await asyncio.sleep(self.step_interval)

            self.state.running = False
            print("[Fake VLA] 运行结束，返回 done")
            await self.broadcast({
                "type": "done",
                "task": task,
                "running": False,
                "success": True,
                "message": "假模型运行完成",
            })
        except Exception as exc:
            logging.exception("Fake VLA task failed")
            self.state.running = False
            await self.broadcast({
                "type": "error",
                "task": task,
                "running": False,
                "success": False,
                "error": str(exc),
                "message": "假模型任务执行失败",
            })
        finally:
            self.state.task_runner = None
            self.state.stop_event.clear()

    async def handle_command(self, message: str) -> None:
        try:
            command = json.loads(message)
        except json.JSONDecodeError:
            await self.broadcast({
                "type": "error",
                "running": self.state.running,
                "success": False,
                "error": "Invalid JSON",
                "message": "消息不是合法 JSON",
            })
            return

        cmd = command.get("cmd")
        if cmd == "status":
            await self.broadcast(self.status_payload(
                "fake model running" if self.state.running else "fake model idle"
            ))
            return

        if cmd == "stop":
            if self.state.running:
                self.state.stop_event.set()
            else:
                await self.broadcast({
                    "type": "stopped",
                    "task": self.state.task,
                    "running": False,
                    "success": False,
                    "message": "当前没有运行中的任务",
                })
            return

        if cmd != "run":
            await self.broadcast({
                "type": "error",
                "task": self.state.task,
                "running": self.state.running,
                "success": False,
                "error": f"Unsupported cmd: {cmd}",
                "message": "不支持的指令",
            })
            return

        task = command.get("task")
        if task not in VALID_TASKS:
            await self.broadcast({
                "type": "error",
                "task": task,
                "running": self.state.running,
                "success": False,
                "error": f"Unsupported task: {task}",
                "message": "不支持的 task",
            })
            return

        async with self.lock:
            if self.state.running:
                await self.broadcast({
                    "type": "error",
                    "task": task,
                    "running": True,
                    "success": False,
                    "error": "Another task is already running",
                    "message": "已有任务在运行",
                })
                return

            self.state.running = True
            self.state.task = task
            self.state.step = 0
            self.state.stop_event.clear()
            self.state.task_runner = asyncio.create_task(self.run_task(task))

    async def websocket_handler(self, websocket, path=None) -> None:
        peer = getattr(websocket, "remote_address", None)
        print(f"[Fake VLA] 客户端连接：{peer}")
        self.state.clients.add(websocket)
        try:
            await websocket.send(json.dumps(
                self.status_payload("fake VLA service connected"),
                ensure_ascii=False,
            ))
            async for message in websocket:
                print(f"[Fake VLA] 收到指令：{message}")
                await self.handle_command(message)
        except ConnectionClosed:
            pass
        finally:
            self.state.clients.discard(websocket)
            print(f"[Fake VLA] 客户端断开：{peer}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fake VLA WebSocket service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=10,
                        help="每个 run 指令模拟执行的 step 数")
    parser.add_argument("--step-interval", type=float, default=0.5,
                        help="每个 step 的间隔秒数")
    parser.add_argument("--status-every-steps", type=int, default=1,
                        help="每隔多少 step 推送一次 status")
    args = parser.parse_args()

    if args.steps <= 0 or args.step_interval < 0 or args.status_every_steps <= 0:
        raise ValueError("steps、step-interval、status-every-steps 必须为正数")

    service = FakeVLAService(args.steps, args.step_interval, args.status_every_steps)
    stop_signal = asyncio.Future()

    def request_shutdown() -> None:
        if not stop_signal.done():
            stop_signal.set_result(None)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except NotImplementedError:
            pass

    print(f"[Fake VLA] 服务启动：ws://{args.host}:{args.port}")
    print("[Fake VLA] 不调用真实模型，不向机器人发送动作。")

    async with websockets.serve(service.websocket_handler, args.host, args.port):
        await stop_signal

    if service.state.running:
        service.state.stop_event.set()
        if service.state.task_runner is not None:
            await service.state.task_runner


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
