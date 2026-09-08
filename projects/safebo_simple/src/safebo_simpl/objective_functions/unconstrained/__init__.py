try:
    from .Ackley import Ackley
    from .Rastrign import Rastrigin
    from .Rosenbrock import Rosenbrock
    from .Sphere import Sphere
except Exception as e:
    raise ImportError(f"Failed to import objective functions! {e}")