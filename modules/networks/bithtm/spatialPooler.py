import numpy as np
from torch import nn

from modules.networks.bithtm.projections import DenseProjection, SparseProjection
from modules.networks.bithtm.regularizations import GlobalInhibition, ExponentialBoosting, UpperLowerThresholdBoosting, \
    LocalInhibition


class SpatialPooler(nn.Module):
    class State:
        def __init__(self, active_columns, overlaps=None, boosted_overlaps=None):
            self.active_columns = active_columns
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps

        def get_bits(self):
            output = np.zeros_like(self.overlaps)
            output[self.active_columns] = 1
            return output

    def __init__(
            self, input_dim, column_dim, sparsity,
            proximal_projection=None, boosting=None, inhibition=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.column_dim = column_dim
        self.sparsity = sparsity

        self.proximal_projection = proximal_projection or DenseProjection(input_dim, column_dim)
        self.boosting = boosting or ExponentialBoosting(column_dim, sparsity)
        self.inhibition = inhibition or GlobalInhibition(sparsity)

        self.state = self.State([])

    def forward(self, input, learning=True):
        overlaps = self.proximal_projection.process(input)
        boosted_overlaps = self.boosting.process(overlaps)
        active_columns = self.inhibition.process(boosted_overlaps)

        if learning:
            self.proximal_projection.update(input, active_columns)
            self.boosting.update(active_columns)

        self.state = self.State(active_columns, overlaps=overlaps, boosted_overlaps=boosted_overlaps)
        return self.state