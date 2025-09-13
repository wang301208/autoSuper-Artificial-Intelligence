from __future__ import annotations

"""Shared global workspace for broadcasting state between modules."""

from typing import Any, Dict, Callable, List, Sequence, Tuple, Optional


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

    # ------------------------------------------------------------------
    def register_module(self, name: str, module: Any) -> None:
        """Register *module* under *name* in the workspace."""
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

        self._state[sender] = state

        att_list: Optional[List[float]] = None
        if attention is not None:
            if isinstance(attention, Sequence) and not isinstance(attention, (str, bytes)):
                att_list = [float(a) for a in attention]
            else:
                att_list = [float(attention)]
            self._attention[sender] = att_list

        # Determine recipients
        recipients = [name for name in self._modules if name != sender]
        if strategy == "local":
            if targets is None:
                raise ValueError("targets must be provided for local strategy")
            recipients = [n for n in targets if n in self._modules and n != sender]
        elif strategy == "sparse":
            if k is None:
                raise ValueError("k must be provided for sparse strategy")
            scores = {
                n: sum(self._attention.get(n, [])) for n in recipients
            }
            recipients = [n for n, _ in sorted(scores.items(), key=lambda i: i[1], reverse=True)[:k]]
        elif targets is not None:
            recipients = [n for n in targets if n in self._modules and n != sender]

        for name in recipients:
            module = self._modules[name]
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                handler(sender, state, att_list)

        for handler in self._state_subs.get(sender, []):
            handler(state)

        if _allow_cross:
            self._trigger_cross_attention(sender)

    def subscribe_state(self, name: str, handler: Callable[[Any], None]) -> None:
        """Invoke *handler* whenever *name* publishes new state."""

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
        self._cross_attn[key] = handler

    def _trigger_cross_attention(self, sender: str) -> None:
        for (a, b), handler in self._cross_attn.items():
            if sender not in (a, b):
                continue
            other = b if sender == a else a
            if other not in self._state:
                continue
            state_a = self._state[a]
            state_b = self._state[b]
            att_a = self._attention.get(a)
            att_b = self._attention.get(b)
            fused_state, fused_attn = handler(state_a, state_b, att_a, att_b)
            self.broadcast(f"{a}|{b}", fused_state, fused_attn, _allow_cross=False)

    # ------------------------------------------------------------------
    def state(self, name: str) -> Any:
        """Return the last state published by *name*."""
        return self._state.get(name)

    def attention(self, name: str) -> Optional[List[float]]:
        """Return the last attention vector published by *name*."""
        return self._attention.get(name)


# Global workspace instance

global_workspace = GlobalWorkspace()
