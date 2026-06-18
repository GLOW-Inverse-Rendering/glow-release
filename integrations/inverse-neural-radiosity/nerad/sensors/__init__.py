from typing import Callable, TypeVar

import mitsuba as mi

from mytorch.registry import import_children

T = TypeVar("T")
registered_sensors = []


def register_sensor(name: str) -> Callable[[T], T]:
    def register(cls: T):
        registered_sensors.append(name)
        mi.register_sensor(name, cls)
        return cls
    return register


import_children(__file__, __name__)
