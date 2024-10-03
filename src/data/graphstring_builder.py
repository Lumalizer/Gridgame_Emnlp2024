from itertools import combinations, combinations_with_replacement, permutations, chain
from dataclasses import dataclass, field
import os
import random


@dataclass
class GraphstringBuilder:
    size: int = 2
    shapes_per_grid: int = 2
    include_lower_shape_amounts: bool = False
    shapes: list[str] = field(default_factory=lambda: [s.replace(".png", "")
                              for s in os.listdir('assets/shapes') if s.endswith('.png')])
    _datastrings: list[str] = field(default_factory=list)

    def __post_init__(self):
        assert (self.size**2 >= self.shapes_per_grid)
        if not os.path.isdir('assets/output'):
            os.mkdir('assets/output')

    def get_shape_combinations(self):
        c = []
        start = 1 if self.include_lower_shape_amounts else self.shapes_per_grid

        for i in range(start, self.shapes_per_grid+1):
            c.extend(list(combinations(self.shapes, i)))
        return c

    def get_grid_items(self, *shapes):
        zeroes = tuple(0 for _ in range(self.size**2 - len(shapes)))
        return zeroes + shapes

    def get_grid_possibilities(self):
        return set(chain.from_iterable(permutations(
            self.get_grid_items(*shapes), self.size**2) for shapes in self.get_shape_combinations()))

    @property
    def datastrings(self):
        if not self._datastrings:
            self._datastrings = sorted(['_'.join([str(p) for p in possibility])
                                       for possibility in self.get_grid_possibilities()])
            random.shuffle(self._datastrings)
        return self._datastrings
