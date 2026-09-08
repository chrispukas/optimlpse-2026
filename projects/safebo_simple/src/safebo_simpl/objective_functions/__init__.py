
try:
    from ._parent import ObjectiveFunction
except Exception as e:
    raise ImportError(f"Failed to objective functions! {e}")