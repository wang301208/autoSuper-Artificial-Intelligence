from __future__ import annotations

"""Shared global workspace for broadcasting state between modules."""

import asyncio
import inspect
import threading
import heapq
from typing import Any, Dict, Callable, List, Sequence, Tuple, Optional, Set


class GlobalWorkspace:
    """Registry that enables modules to share state and attention."""

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}
        # Store per-module multi-head attention weights
        self._attention: Dict[str, List[float]] = {}
        self._state_subs: Dict[str, List[Callable[[Any], None]]] = {}
        # (module_a, module_b) -> cross-attention fusion hook
        self._cross_attn: Dict[Tuple[str, str], Callable[[Any, Any, Optional[List[float]], Optional[List[float]]], Tuple[Any, Optional[List[float]]]]] = {}
        # Synchronization lock for internal dictionaries
        self._lock = threading.RLock()
        # Candidate queue sorted by activation (negative for max-heap)
        self._candidates: List[Tuple[float, int, Tuple[str, Any, Optional[List[float]], str, Optional[List[str]], Optional[int], bool]]] = []
        self._candidate_counter = 0
        self._attention_threshold = 0.0

    # ------------------------------------------------------------------
    def register_module(self, name: str, module: Any) -> None:
        """Register *module* under *name* in the workspace."""
        with self._lock:
            self._modules[name] = module

    def broadcast(
        self,
        sender: str,
        state: Any,
        attention: Optional[Sequence[float] | float] = None,
        *,
        strategy: str = "full",
        targets: Optional[List[str]] = None,
        k: Optional[int] = None,
        _allow_cross: bool = True,
    ) -> None:
        """Broadcast *state* from *sender* according to *strategy*.

        Parameters
        ----------
        sender:
            Name of the broadcasting module.
        state:
            Arbitrary payload to share.
        attention:
            Optional attention vector. A scalar will be promoted to a
            single-element list.
        strategy:
            ``"full"`` (default) broadcasts to all modules, ``"local"`` only to
            provided ``targets`` and ``"sparse"`` routes to the top ``k`` modules
            ranked by their current attention weights.
        targets:
            Optional explicit list of recipient modules. Required for
            ``strategy="local"``.
        k:
            Number of modules to target when ``strategy="sparse"``.
        _allow_cross:
            Internal flag to prevent recursive cross attention broadcasting.
        """

        with self._lock:
            self._state[sender] = state

            att_list: Optional[List[float]] = None
            if attention is not None:
                if isinstance(attention, Sequence) and not isinstance(attention, (str, bytes)):
                    att_list = [float(a) for a in attention]
                else:
                    att_list = [float(attention)]
                self._attention[sender] = att_list

            activation = sum(att_list) if att_list else 0.0
            data = (sender, state, att_list, strategy, targets, k, _allow_cross)
            heapq.heappush(self._candidates, (-activation, self._candidate_counter, data))
            self._candidate_counter += 1

        self._process_queue()

    def subscribe_state(self, name: str, handler: Callable[[Any], None]) -> None:
        """Invoke *handler* whenever *name* publishes new state."""

        with self._lock:
            self._state_subs.setdefault(name, []).append(handler)

    def register_cross_attention(
        self,
        module_a: str,
        module_b: str,
        handler: Callable[[Any, Any, Optional[List[float]], Optional[List[float]]], Tuple[Any, Optional[List[float]]]],
    ) -> None:
        """Register a cross-attention fusion hook between two modules.

        When both modules have published state, *handler* is invoked with
        ``(state_a, state_b, attn_a, attn_b)`` and should return a tuple of
        ``(fused_state, fused_attention)`` which is then broadcast under the
        sender name ``"module_a|module_b"``.
        """

        key = tuple(sorted((module_a, module_b)))
        with self._lock:
            self._cross_attn[key] = handler

    def _trigger_cross_attention(self, sender: str) -> None:
        with self._lock:
            items = list(self._cross_attn.items())
            states = dict(self._state)
            attns = dict(self._attention)

        for (a, b), handler in items:
            if sender not in (a, b):
                continue
            other = b if sender == a else a
            if other not in states:
                continue
            state_a = states[a]
            state_b = states[b]
            att_a = attns.get(a)
            att_b = attns.get(b)
            fused_state, fused_attn = handler(state_a, state_b, att_a, att_b)
            self.broadcast(f"{a}|{b}", fused_state, fused_attn, _allow_cross=False)

    # ------------------------------------------------------------------
    def state(self, name: str) -> Any:
        """Return the last state published by *name*."""
        with self._lock:
            return self._state.get(name)

    def attention(self, name: str) -> Optional[List[float]]:
        """Return the last attention vector published by *name*."""
        with self._lock:
            return self._attention.get(name)

    # ------------------------------------------------------------------
    def set_attention_threshold(self, threshold: float) -> None:
        """Set the activation threshold required to enter consciousness."""

        with self._lock:
            self._attention_threshold = float(threshold)

    # Internal methods --------------------------------------------------
    def _process_queue(self) -> None:
        while True:
            with self._lock:
                if not self._candidates:
                    return
                activation_neg, _, data = self._candidates[0]
                activation = -activation_neg
                if activation < self._attention_threshold:
                    return
                heapq.heappop(self._candidates)
            sender, state, att_list, strategy, targets, k, allow_cross = data
            recipients = self._deliver_broadcast(sender, state, att_list, strategy, targets, k, allow_cross)
            self._push_to_all(sender, state, att_list, recipients)

    def _deliver_broadcast(
        self,
        sender: str,
        state: Any,
        att_list: Optional[List[float]],
        strategy: str,
        targets: Optional[List[str]],
        k: Optional[int],
        allow_cross: bool,
    ) -> Set[str]:
        """Deliver broadcast to recipients based on strategy.

        Returns the set of recipient module names.
        """

        with self._lock:
            recipients = [name for name in self._modules if name != sender]
            if strategy == "local":
                if targets is None:
                    raise ValueError("targets must be provided for local strategy")
                recipients = [n for n in targets if n in self._modules and n != sender]
            elif strategy == "sparse":
                if k is None:
                    raise ValueError("k must be provided for sparse strategy")
                scores = {n: sum(self._attention.get(n, [])) for n in recipients}
                recipients = [n for n, _ in sorted(scores.items(), key=lambda i: i[1], reverse=True)[:k]]
            elif targets is not None:
                recipients = [n for n in targets if n in self._modules and n != sender]

            modules = {name: self._modules[name] for name in recipients}
            subs = list(self._state_subs.get(sender, []))

        for name, module in modules.items():
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(sender, state, att_list))
                else:
                    handler(sender, state, att_list)

        for handler in subs:
            handler(state)

        if allow_cross:
            self._trigger_cross_attention(sender)

        return set(modules.keys())

    def _push_to_all(
        self,
        sender: str,
        state: Any,
        att_list: Optional[List[float]],
        already_sent: Set[str],
    ) -> None:
        """After broadcast, push the state to all remaining modules."""

        with self._lock:
            recipients = [n for n in self._modules if n not in already_sent and n != sender]
            modules = {name: self._modules[name] for name in recipients}

        for module in modules.values():
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(sender, state, att_list))
                else:
                    handler(sender, state, att_list)


# Global workspace instance

global_workspace = GlobalWorkspace()
