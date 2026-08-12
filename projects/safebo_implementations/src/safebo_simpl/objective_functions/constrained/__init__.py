try:
    from .KeanesBump import KeanesBump
except:
    raise ValueError(f"Failed to import {__name__} classes!")