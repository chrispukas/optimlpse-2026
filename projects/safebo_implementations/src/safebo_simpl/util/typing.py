from typing import Any


type Conditional[T] = T | None

def _factory[T](
        val: T | type[T],
        **kwargs: Any
        ) -> T:
    """
        Factory method to allow arguments to accept both types, and instantiated class objects
    """
    return val(**kwargs) if isinstance(val, type) else val # type: ignore
