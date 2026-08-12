from typing import Any, Callable


type AllowUndefined[T] = T | None
type InitializerCallable[A, T] = AllowUndefined[Callable[[A], T]]

def _factory[T](
        val: T | type[T],
        **kwargs: Any
        ) -> T:
    """
        Factory method to allow arguments to accept both types, and instantiated class objects
    """
    return val(**kwargs) if isinstance(val, type) else val # type: ignore
