from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Union

import numpy as np

from ...layers.utils.color_transformations import (
    transform_color,
    transform_color_cycle,
)
from ...utils.events.custom_types import Array
from ..translations import trans


@dataclass(eq=False)
class ColorCycle:
    """A dataclass to hold a color cycle for the fallback_colors
    in the CategoricalColormap

    Attributes
    ----------
    values : np.ndarray
        The (Nx4) color array of all colors contained in the color cycle.
    cycle : cycle
        The cycle object that gives fallback colors.
    """

    values: np.ndarray
    cycle: cycle

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        # turn a generic dict into object
        if isinstance(val, dict):
            return _coerce_colorcycle_from_dict(val)
        elif isinstance(val, ColorCycle):
            return val
        else:
            return _coerce_colorcycle_from_colors(val)

    def _json_encode(self):
        return {'values': self.values.tolist()}

    def __eq__(self, other):
        if isinstance(other, ColorCycle):
            eq = np.array_equal(self.values, other.values)
        else:
            eq = False
        return eq


def _coerce_colorcycle_from_dict(
    val: Dict[str, Union[str, list, np.ndarray, cycle]]
) -> ColorCycle:
    # validate values
    color_values = val.get('values')
    if color_values is None:
        raise ValueError(
            trans._('ColorCycle requires a values argument', deferred=True)
        )

    transformed_color_values = transform_color(color_values)

    # validate cycle
    color_cycle = val.get('cycle')
    if color_cycle is None:
        transformed_color_cycle = transform_color_cycle(
            color_cycle=color_values,
            elem_name='color_cycle',
            default="white",
        )[0]
    else:
        transformed_color_cycle = color_cycle

    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def _coerce_colorcycle_from_colors(
    val: Union[str, list, np.ndarray]
) -> ColorCycle:
    if isinstance(val, str):
        val = [val]
    (
        transformed_color_cycle,
        transformed_color_values,
    ) = transform_color_cycle(
        color_cycle=val,
        elem_name='color_cycle',
        default="white",
    )
    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def compare_colormap_dicts(cmap_1, cmap_2):

    if len(cmap_1) != len(cmap_2):
        return False
    for k, v in cmap_1.items():
        if k not in cmap_2:
            return False
        if not np.allclose(v, cmap_2[k]):
            return False
    return True


def _map_dictionary(
    dictionary: Dict, keys: Array, dict_size_cutoff: int = 100
) -> Array:
    """optimized version of dictionary lookup via numpy

    Parameters
    ----------
    dictionary : Dict
        dictionary mapping keys to values
    keys : Array
        the array of keys
    dict_size_cutoff : int, optional
        cutoff size for dictionary above which we will fall back to non-optimized keyval mapping
        (as the optimized version will create a temporary array of size len(dictionary.keys()) * len(keys)
        which might be too costly, 100 should be a reasonable default)

    Example
    --------

    d = dict((i,i) for i in range(5))
    x = np.random.randint(0,5, 10**6)

    %timeit [d[_x] for _x in x]
    721 ms ± 43.4 ms per loop

    %timeit _map_dictionary(d,x)
    79.7 ms ± 3.48 ms per loop

    """

    keys = np.asanyarray(keys)

    if len(dictionary.keys()) <= dict_size_cutoff:
        # optimized version cf https://stackoverflow.com/a/28984192
        dict_keys = np.array(tuple(dictionary.keys()))
        vals = np.array(tuple(dictionary.values()))
        result = vals[(np.atleast_2d(keys).T == dict_keys).argmax(axis=1)]
    else:
        # fall back version (loop over values.tolist(), cf https://stackoverflow.com/a/24706209 )
        result = np.array([dictionary[x] for x in keys.tolist()])

    return result
