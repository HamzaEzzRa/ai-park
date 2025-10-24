# Lightweight FSM implementation derived from 'transitions' library
# https://github.com/pytransitions/transitions

from __future__ import annotations

from collections import defaultdict
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from graphviz import Digraph

Callback = Callable[["EventData"], Any]
Condition = Callable[["EventData"], bool]


class MachineError(RuntimeError):
    """Raised when the state machine fails to transition."""


class State:
    """Container for state metadata and lifecycle callbacks."""

    def __init__(
        self,
        name: str | Enum,
        *,
        on_enter: Optional[Callback | Sequence[Callback]] = None,
        on_exit: Optional[Callback | Sequence[Callback]] = None,
    ) -> None:
        self._name = name
        self._on_enter = self._listify_callbacks(on_enter)
        self._on_exit = self._listify_callbacks(on_exit)

    @staticmethod
    def _listify_callbacks(value: Optional[Callback | Sequence[Callback]]) -> List[Callback]:
        if value is None:
            return []
        if callable(value):
            return [value]
        return [cb for cb in value if callable(cb)]

    @property
    def name(self) -> str:
        return self._name.name if isinstance(self._name, Enum) else str(self._name)

    @property
    def value(self) -> Any:
        return self._name.value if isinstance(self._name, Enum) else self._name

    def enter(self, data: "EventData") -> None:
        data.machine._invoke_callbacks(self._on_enter, data)

    def exit(self, data: "EventData") -> None:
        data.machine._invoke_callbacks(self._on_exit, data)


class EventData:
    """Event context shared by callbacks and transition logic."""

    def __init__(
        self,
        machine: "Machine",
        event: Optional["Event"],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.machine = machine
        self.event = event
        self.args = args
        self.kwargs = kwargs or {}
        self.transition: Optional["Transition"] = None
        self.source: Optional[State] = machine.current_state
        self.dest: Optional[State] = None
        self.result: bool = False


class Transition:
    """Directed edge between two states."""

    def __init__(
        self,
        source: str,
        dest: str,
        *,
        conditions: Optional[Sequence[Condition] | Condition] = None,
        before: Optional[Callback | Sequence[Callback]] = None,
        after: Optional[Callback | Sequence[Callback]] = None,
    ) -> None:
        self.source = source
        self.dest = dest
        if conditions is None:
            self.conditions: List[Condition] = []
        elif callable(conditions):
            self.conditions = [conditions]
        else:
            self.conditions = list(conditions)
        self.before = State._listify_callbacks(before)
        self.after = State._listify_callbacks(after)

    def execute(self, data: EventData) -> bool:
        data.transition = self
        data.dest = data.machine.get_state(self.dest)

        for guard in self.conditions:
            if not guard(data):
                return False

        data.machine._invoke_callbacks(self.before, data)
        data.machine._change_state(data.dest, data)
        data.machine._invoke_callbacks(self.after, data)
        data.result = True
        return True


class Event:
    """Groups transitions for a trigger name."""

    def __init__(self, name: str, machine: "Machine") -> None:
        self.name = name
        self.machine = machine
        self._transitions: Dict[str, List[Transition]] = defaultdict(list)

    def add_transition(self, transition: Transition) -> None:
        self._transitions[transition.source].append(transition)

    def fire(self, *args: Any, **kwargs: Any) -> bool:
        machine = self.machine
        current = machine.current_state
        data = EventData(machine, self, args, kwargs)
        if current is None:
            return False

        for key in (current.name, Machine.wildcard_symbol):
            for transition in self._transitions.get(key, []):
                if transition.execute(data):
                    return True
        return False


class Machine:
    """Deterministic finite state machine with auto-generated triggers."""

    wildcard_symbol = "*"

    def __init__(
        self,
        states: Optional[Sequence[str | Enum | State]] = None,
        initial_state: Optional[str | Enum] = None
    ) -> None:
        if initial_state is not None:
            if states is None or initial_state not in states:
                raise ValueError("Initial state must be in the provided list of states.")

        self._states: Dict[str, State] = {}
        self._events: Dict[str, Event] = {}
        self._current_state: Optional[State] = None
        self._initial_state: Optional[State] = None

        if states:
            self.add_states(states)
        if initial_state is not None:
            self._initial_state = self.get_state(initial_state)
            self.set_state(self._initial_state.name)

    @property
    def states(self) -> Dict[str, State]:
        return dict(self._states)

    @property
    def current_state(self) -> Optional[State]:
        return self._current_state

    @property
    def initial_state(self) -> Optional[State]:
        return self._initial_state

    def add_state(
        self,
        state: str | Enum | State,
        *,
        on_enter: Optional[Callback | Sequence[Callback]] = None,
        on_exit: Optional[Callback | Sequence[Callback]] = None,
    ) -> None:
        if isinstance(state, State):
            name = state.name
            self._states[name] = state
        else:
            name = state.name if isinstance(state, Enum) else str(state)
            if name in self._states:
                raise ValueError(f"State '{name}' already registered.")

            state = state if isinstance(state, Enum) else str(state)
            self._states[name] = State(state, on_enter=on_enter, on_exit=on_exit)

    def add_states(
        self,
        states: Sequence[str | Enum | State],
        *,
        on_enter: Optional[Sequence[Callback | Sequence[Callback]]] = None,
        on_exit: Optional[Sequence[Callback | Sequence[Callback]]] = None,
    ) -> None:
        on_enter = on_enter or []
        on_exit = on_exit or []
        for idx, state in enumerate(states):
            enter_cb = on_enter[idx] if idx < len(on_enter) else None
            exit_cb = on_exit[idx] if idx < len(on_exit) else None
            self.add_state(state, on_enter=enter_cb, on_exit=exit_cb)

    def get_state(self, name: str | Enum) -> State:
        key = name.name if isinstance(name, Enum) else str(name)
        if key not in self._states:
            raise ValueError(f"State '{key}' not found in machine states.")
        return self._states[key]

    def set_state(self, name: str | Enum) -> None:
        dest = self.get_state(name)
        data = EventData(self, None)
        data.dest = dest
        self._change_state(dest, data)

    def reset(self) -> None:
        if self._initial_state is None:
            self._current_state = None
            return
        self.set_state(self._initial_state.name)

    def add_transition(
        self,
        sources: str | Enum | Sequence[str | Enum] | str,
        dest: str | Enum,
        trigger: str | Enum,
        *,
        conditions: Optional[Sequence[Condition] | Condition] = None,
        before: Optional[Callback | Sequence[Callback]] = None,
        after: Optional[Callback | Sequence[Callback]] = None,
    ) -> None:
        if isinstance(trigger, IntFlag):
            mask = trigger.value or 0
            trigger_name = self._trigger_label(mask, trigger.__class__) if mask else ""
        else:
            trigger_name = (
                trigger.name.lower() if isinstance(trigger, Enum)
                else str(trigger)
            )

        if sources == self.wildcard_symbol:
            source_names = list(self._states.keys())
        elif isinstance(sources, (list, tuple)):
            source_names = [s.name if isinstance(s, Enum) else str(s) for s in sources]
        else:
            source_names = [sources.name if isinstance(sources, Enum) else str(sources)]

        dest_name = dest.name if isinstance(dest, Enum) else str(dest)
        if dest_name not in self._states:
            raise ValueError(f"Unknown destination state '{dest_name}'.")

        event = self._ensure_event(trigger_name)
        for src in source_names:
            if src != self.wildcard_symbol and src not in self._states:
                raise ValueError(f"Unknown source state '{src}'.")
            transition = Transition(
                source=src,
                dest=dest_name,
                conditions=conditions,
                before=before,
                after=after,
            )
            event.add_transition(transition)

    def trigger(self, name: str | Enum, *args: Any, **kwargs: Any) -> bool:
        if isinstance(name, IntFlag):
            mask = name.value or 0
            name = self._trigger_label(mask, name.__class__) if mask else ""
        else:
            name = (
                name.name.lower() if isinstance(name, Enum)
                else str(name)
            )
        if name not in self._events:
            return False
        event = self._events[name]
        if event.fire(*args, **kwargs):
            return True
        return False

    def to_graphviz(self) -> Digraph:
        g = Digraph()
        for state in self.states.values():
            shape = "doublecircle" if state is self.current_state else "circle"
            g.node(state.name, shape=shape)

        for trigger, event in self._events.items():
            trigger = trigger.replace("_and_", " & ")
            for src, transitions in event._transitions.items():
                for tr in transitions:
                    g.edge(src, tr.dest, label=trigger)
        return g

    def _ensure_event(self, name: str | Enum) -> Event:
        if isinstance(name, IntFlag):
            mask = name.value or 0
            trigger_name = self._trigger_label(mask, name.__class__) if mask else ""
        else:
            trigger_name = (
                name.name.lower() if isinstance(name, Enum)
                else str(name)
            )
        if trigger_name not in self._events:
            event = Event(trigger_name, self)
            self._events[trigger_name] = event

            if trigger_name != "":
                def trigger_method(*args: Any, _event: Event = event, **kwargs: Any) -> bool:
                    if _event.fire(*args, **kwargs):
                        return True
                    return False
                setattr(self, trigger_name, trigger_method)
        return self._events[trigger_name]

    def _change_state(self, dest: State, data: EventData) -> None:
        previous = self._current_state
        if previous is dest:
            return
        data.source = previous
        if previous is not None:
            previous.exit(data)
        self._current_state = dest
        dest.enter(data)

    def _invoke_callbacks(self, callbacks: Iterable[Callback], data: EventData) -> None:
        for callback in callbacks:
            callback(data)

    def _trigger_label(self, mask: int, trigger_cls) -> str:
        labels = []
        if isinstance(trigger_cls, type) and issubclass(trigger_cls, Enum):
            for bit in trigger_cls:
                if mask & bit.value:
                    labels.append(bit.name.lower())
        if not labels:
            labels.append(str(mask))
        return "_and_".join(labels).lower()
