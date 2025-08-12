import asyncio
from typing import Dict, Any, Optional, Set


class EvaluationProgressManager:
  def __init__(self) -> None:
    self._queues: Dict[str, asyncio.Queue] = {}
    self._results: Dict[str, Dict[str, Any]] = {}
    self._done: Dict[str, bool] = {}
    self._listeners: Dict[str, Set[Any]] = {}

  def create(self, trace_id: str) -> None:
    if trace_id not in self._queues:
      self._queues[trace_id] = asyncio.Queue()
      self._listeners[trace_id] = set()
      self._done[trace_id] = False

  def is_done(self, trace_id: str) -> bool:
    return self._done.get(trace_id, False)

  def set_result(self, trace_id: str, result: Dict[str, Any]) -> None:
    self._results[trace_id] = result
    self._done[trace_id] = True

  def get_result(self, trace_id: str) -> Optional[Dict[str, Any]]:
    return self._results.get(trace_id)

  async def publish(self, trace_id: str, stage: str, message: str = "", percent: Optional[float] = None, data: Optional[Dict[str, Any]] = None) -> None:
    if trace_id not in self._queues:
      self.create(trace_id)
    payload = {"stage": stage, "message": message, "percent": percent, "data": data}
    await self._queues[trace_id].put(payload)
    # push to active listeners
    listeners = list(self._listeners.get(trace_id, set()))
    for ws in listeners:
      try:
        await ws.send_json({"type": "progress", **payload})
      except Exception:
        try:
          self._listeners[trace_id].discard(ws)
        except Exception:
          pass

  async def stream(self, trace_id: str):
    if trace_id not in self._queues:
      self.create(trace_id)
    queue = self._queues[trace_id]
    while not self._done.get(trace_id, False):
      item = await queue.get()
      yield item

  def attach(self, trace_id: str, ws) -> None:
    if trace_id not in self._listeners:
      self._listeners[trace_id] = set()
    self._listeners[trace_id].add(ws)

  def detach(self, trace_id: str, ws) -> None:
    try:
      self._listeners.get(trace_id, set()).discard(ws)
    except Exception:
      pass


progress_manager = EvaluationProgressManager()


