"""
Part of the H9 project
"""
from typing import Optional
import numpy as np
from .domain import Domain


class Points(np.ndarray):
    """
    A domain-aware collection of coordinate positions.
    Each 'point' represents a location that may be approximate,
    depending on its Domain and formatting resolution.

    Subclass of numpy.ndarray that supports a Domain context.
    Address formats are defined by the Domain this belongs to.
    """
    dom: Optional[Domain]

    def __new__(cls, input_array, sys='wa'):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.dom = None  # Store domain sig.
        return obj

    def set_domain(self, _set: Domain):
        """Set the domain for access to its formatters."""
        self.dom = _set
        return self

    def domain(self):
        """Return domain of current points."""
        return self.dom.name if self.dom else None

    def __array_finalize__(self, obj):
        """ Used in copying, and via functions."""
        if obj is None:
            return
        self.dom = getattr(obj, 'dom', None)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if isinstance(result, np.ndarray):
            result = result.view(Points)
            result.dom = self.dom
        return result

    def __format__(self, format_spec):
        """Allow f-string formatting."""
        if self.dom is not None and format_spec is not None and format_spec != '':
            main_sub = format_spec.split('.')
            name = main_sub[0]
            sub = main_sub[1] if len(main_sub) > 1 else ''
            if name not in self.dom.address_formats:
                raise ValueError(f"Unknown format '{name}' for {self.dom.name}")
            formatter = self.dom.address_formats[name]
            return formatter.format(self, sub)
        return super().__format__('')

    def __repr__(self):
        base = super().__repr__()
        domain = self.dom.name if self.dom else 'None'
        return f"{base}, domain='{domain}'"

    def __bool__(self):
        return self is not None and super().__len__() > 0

    def __len__(self):
        return super().__len__()
