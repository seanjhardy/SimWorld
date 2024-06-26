# SOURCE: adapted from https://github.com/cokwa/bitHTM
from torch import nn

from .regularizations import LocalInhibition, GlobalInhibition
from .spatialPooler import SpatialPooler
from .temporalMemory import TemporalMemory
import numpy as np


class HierarchicalTemporalMemory(nn.Module):
    def __init__(
        self, input_size, columns, cells, apical_input_size=0, sparsity=0.02,
            spatial_pooler=None, temporal_memory=None, local=False):
        super().__init__()
        self.input_size = input_size
        self.columns = columns
        self.cells = cells
        self.sparsity = sparsity

        self.models = nn.ModuleDict(dict(
            identity=nn.Identity(),
            spatial_pooler=spatial_pooler or SpatialPooler(input_size, columns, sparsity,
                                                           inhibition=LocalInhibition(sparsity) if local else None),
            temporal_memory=temporal_memory or TemporalMemory(columns, cells, apical_dim=apical_input_size)
        ))

    def run(self, input, apical_input=None):
        self.models.identity(input)
        # prev_predicted = self.temporal_memory.last_state.cell_prediction.reshape(-1)

        sp_state = self.models.spatial_pooler(input, learning=True)
        tm_state = self.models.temporal_memory(sp_state, apical_input, learning=True)

        tm_pred = np.array(tm_state.cell_prediction.max(axis=1), dtype=np.int32)
        
        activity = np.zeros(self.columns, dtype=np.bool_) 
        activity[sp_state.active_columns] = True

        return sp_state, tm_pred
    
    def inverse(self, sdr, inhibition=None, targets=None):
        output = np.zeros(self.input_size)
        predictions = self.models.spatial_pooler.proximal_projection.invert(sdr)

        if inhibition is not None:
            predictions = inhibition.process(predictions)
        #else:
            #inhib = GlobalInhibition(0.1)
            #predictions = inhib.process(predictions)

        output[np.where(predictions > 0)] = 1

        if targets is not None:
            diff = output - targets
            self.models.spatial_pooler.proximal_projection.permanence -= np.outer(sdr, diff) * 0.01

        return output
            
