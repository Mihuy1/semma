"""Results of `DaemonClient` futures, delivered on the Qt thread."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class Relay(QObject):
    """Runs a future's callbacks on the thread this object lives on (the GUI thread).

    The future completes on the client's worker thread; emitting a signal from there to an
    object of the GUI thread is a queued connection, so the callback runs in the Qt event loop.
    Give it a parent widget: once that widget is gone, results still arriving are dropped."""

    _fire = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fire.connect(self._run)

    @Slot(object)
    def _run(self, fn: Callable[[], None]) -> None:
        fn()

    def post(self, fn: Callable[[], None]) -> None:
        """Run `fn` on this object's thread; safe to call from any thread."""
        try:
            self._fire.emit(fn)
        except RuntimeError:  # deleted meanwhile
            pass

    def when_done(self, future: Future[Any], on_result: Callable[[Any], None],
                  on_error: Callable[[BaseException], None] | None = None) -> None:
        def deliver() -> None:
            if future.cancelled():
                return
            exc = future.exception()
            if exc is None:
                on_result(future.result())
            elif on_error is not None:
                on_error(exc)

        future.add_done_callback(lambda _: self.post(deliver))

