from .projections import SparseProjection, DenseProjection, PredictiveProjection
from .regularizations import ExponentialBoosting, GlobalInhibition, LocalInhibition, NoBoosting, SoftmaxInhibition, UpperLowerThresholdBoosting
from .utils import DynamicArray2D
from .HTM import SpatialPooler
import numpy as np

class AdjustedTemporalPooler(SpatialPooler):
    class TPState:
        def __init__(self, active_columns, overlaps=None):
            self.active_columns = active_columns
            self.overlaps = overlaps
            
    def __init__(self, input_dim, column_dim, sparsity=0.02, inhibition=None,
                 activeOverlapWeight=1.0,
                predictedActiveOverlapWeight=2.0,
                maxUnionActivity=0.20,
                synPermPredActiveInc=0.1,
                synPermPreviousPredActiveInc=0.05):
        super().__init__(input_dim, column_dim, sparsity, 
                     proximal_projection=DenseProjection(input_dim, column_dim, permanence_mean=0.1, permanence_std=0.0),
                     boosting=ExponentialBoosting(column_dim, sparsity, intensity=0.5))
        self.column_dim = column_dim
        self.active_overlap_weight = activeOverlapWeight
        self.predicted_active_overlap_weight = predictedActiveOverlapWeight
        
        self.syn_perm_pred_active_inc = synPermPredActiveInc
        self.syn_perm_previous_pred_active_inc = synPermPreviousPredActiveInc
        
        self.active_cells = np.array([], dtype=np.int32)
        
        self.history_length = 5
        self.pre_predicted_active_input = np.zeros((input_dim, self.history_length), dtype=np.int32)
        self.state = self.TPState(np.array([]),[])
        self.column_connections = np.zeros(column_dim)
        self.lengths = np.zeros(column_dim)

    def run(self, active_input, predicted_input, next_predicted_input, learning=True):
        predicted_active_input = active_input & predicted_input
        
        #active_overlap = self.proximal_projection.process(active_input)
        #predicted_overlap = self.proximal_projection.process(predicted_input)
        predicted_active_overlap = self.proximal_projection.process(predicted_active_input)
        next_predicted_overlap = self.proximal_projection.process(next_predicted_input)
        
        # Sum active and predicted cells
        total_overlap = next_predicted_overlap * self.active_overlap_weight + predicted_active_overlap * self.predicted_active_overlap_weight
        
        k = int(self.column_dim * self.sparsity)
        
        if np.count_nonzero(total_overlap) == 0:
            # Randomly pick k least_used cells to represent this input
            active_columns = np.argpartition(self.column_connections + np.random.randn(self.column_dim), k)[:k]#np.array([np.argmin(self.column_connections)])
            self.column_connections[active_columns] += 1 # update connections
            total_overlap = np.zeros(self.column_dim)
            total_overlap[active_columns] += 1
        else:
            boosted_overlap = self.boosting.process(total_overlap)
            #active_columns = np.nonzero(boosted_overlap > 5)[0]
            active_columns = self.inhibition.process(total_overlap)
        
        if learning:
            self.proximal_projection.update(predicted_active_input, active_columns,
                                            permanence_incr=0.1, 
                                            permanence_decr=0.1)
            
            for i in range(self.history_length):
                if np.count_nonzero(self.pre_predicted_active_input[:,i]) == 0:
                    break
                self.proximal_projection.update(self.pre_predicted_active_input[:,0], active_columns,
                                            permanence_incr=self.syn_perm_previous_pred_active_inc/(i+1), 
                                            permanence_decr=0.01)
            
        self.boosting.update(active_columns)
        
        #Save previous inputs
        self.pre_predicted_active_input = np.roll(self.pre_predicted_active_input, 1, 1)
        self.pre_predicted_active_input[:,0] = predicted_active_input

        self.state = self.TPState(active_columns, overlaps=total_overlap)
        
        return self.state
    
class TemporalPooler(SpatialPooler):
    class TPState:
        def __init__(self, active_columns, union_SDR, overlaps=None, boosted_overlaps=None):
            self.active_columns = active_columns
            self.union_SDR = union_SDR
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps
            
    def __init__(self, input_dim, column_dim, sparsity, inhibition=None,
                 activeOverlapWeight=1.0,
                predictedActiveOverlapWeight=10.0,
                maxUnionActivity=0.10,
                synPermPredActiveInc=0.1,
                synPermPreviousPredActiveInc=0.2,
                historyLength=5):
        super().__init__(input_dim, column_dim, sparsity, inhibition=inhibition, boosting=ExponentialBoosting(column_dim, sparsity, intensity=0.01))
        self.active_overlap_weight = activeOverlapWeight
        self.predicted_active_overlap_weight = predictedActiveOverlapWeight
        self.max_union_activity = maxUnionActivity
        
        self.syn_perm_active_inc = 0.2
        self.syn_perm_inactive_dec = 0.1
        self.syn_perm_pred_active_inc = synPermPredActiveInc
        self.syn_perm_previous_pred_active_inc = synPermPreviousPredActiveInc
        self.history_length = historyLength
        self.min_history = 0
        
        self.max_union_cells = int(column_dim * self.max_union_activity)
        self.pooling_activation = np.zeros(column_dim, dtype=np.float32)

        self.pooling_timer = np.ones(column_dim, dtype=np.float32) * 1000
        self.pooling_activation_init_level = np.zeros(column_dim, dtype=np.float32)
        self.pooling_activation_tie_breaker = np.random.randn(column_dim) * 0.00001
        
        self.active_cells = np.array([], dtype=np.int32)

        self.pre_predicted_active_input = np.zeros((input_dim, self.history_length), dtype=np.int32)
        self.state = self.TPState([],[])

    def run(self, active_input, predicted_input, learning=True):
        predicted_active_input = active_input & predicted_input
        active_overlap = self.proximal_projection.process(active_input)
        predicted_overlap = self.proximal_projection.process(predicted_active_input)
        
        total_overlap = active_overlap * self.active_overlap_weight + predicted_overlap * self.predicted_active_overlap_weight
        
        boosted_overlaps = self.boosting.process(total_overlap)
        active_columns = self.inhibition.process(boosted_overlaps)
        
        self.update_pooling(total_overlap, active_columns)
        
        # update union SDR
        union_SDR = self.get_most_active_cells()
        
        if learning:
            self.proximal_projection.update(predicted_active_input, active_columns,
                                            permanence_incr=self.syn_perm_active_inc, 
                                            permanence_decr=self.syn_perm_inactive_dec)
            
            self.proximal_projection.update(predicted_active_input, union_SDR,
                                            permanence_incr=self.syn_perm_pred_active_inc, 
                                            permanence_decr=0.0)
            
            
            for i in range(self.history_length):
                self.proximal_projection.update(self.pre_predicted_active_input[:,i], active_columns,
                                            permanence_incr=self.syn_perm_previous_pred_active_inc, 
                                            permanence_decr=0.0)
            
        self.boosting.update(active_columns)
        
        #Save previous inputs
        self.pre_predicted_active_input = np.roll(self.pre_predicted_active_input, 1, 1)
        if self.history_length > 0:
            self.pre_predicted_active_input[:,0] = predicted_active_input
            
        self.state = self.TPState(active_columns, union_SDR, overlaps=total_overlap, boosted_overlaps=boosted_overlaps)
        
        return self.state
    
    def update_pooling(self, overlaps, active_cells):
        self.pooling_activation = np.exp(-0.1 * self.pooling_timer) *  self.pooling_activation_init_level
        
        # Sigmoid activation
        self.pooling_activation[active_cells] += overlaps[active_cells]
        
        self.pooling_timer[self.pooling_timer >= 0] += 1
        self.pooling_timer[active_cells] = 0
        self.pooling_activation_init_level[active_cells] = self.pooling_activation[active_cells]
        
        return self.pooling_activation
    
    def get_most_active_cells(self):
        top_cells = self.inhibition.process(self.pooling_activation)
                
        if max(self.pooling_timer) > self.min_history:
            union_SDR = np.sort(top_cells).astype(np.int32)
        else:
            union_SDR = []
            
        return union_SDR

class SandwichPooler(SpatialPooler):
    class SState:
        def __init__(self, active_columns, union_SDR, overlaps=None, boosted_overlaps=None):
            self.active_columns = active_columns
            self.union_SDR = union_SDR
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps
            
    def __init__(self, input_dim, column_dim, sparsity, inhibition=None,
                 activeOverlapWeight=1.0,
                predictedActiveOverlapWeight=10.0,
                maxUnionActivity=0.10,
                synPermPredActiveInc=0.1,
                synPermPreviousPredActiveInc=0.2,
                historyLength=5):
        super().__init__(input_dim, column_dim, sparsity, inhibition=inhibition, boosting=ExponentialBoosting(column_dim, sparsity, intensity=0.05))
        self.active_overlap_weight = activeOverlapWeight
        self.predicted_active_overlap_weight = predictedActiveOverlapWeight
        self.max_union_activity = maxUnionActivity
        
        self.proximal_projection_upper = DenseProjection(column_dim, column_dim)
        self.syn_perm_active_inc = 0.2
        self.syn_perm_inactive_dec = 0.1
        self.syn_perm_pred_active_inc = synPermPredActiveInc
        self.syn_perm_previous_pred_active_inc = synPermPreviousPredActiveInc
        self.history_length = historyLength
        self.min_history = 0
        self.pooling_threshold = 0.05
        # Decay rate, such that after history_length steps, the a value of 1 will be pooling_threshold % of the original value
        self.alpha = 1.0 - (self.pooling_threshold) ** (1.0/self.history_length)
        
        self.max_union_cells = int(column_dim * self.max_union_activity)
        self.pooling_activation = np.zeros(column_dim, dtype=np.float32)

        self.pooling_timer = np.ones(column_dim, dtype=np.float32) * 1000
        self.pooling_activation_init_level = np.zeros(column_dim, dtype=np.float32)
        self.pooling_activation_tie_breaker = np.random.randn(column_dim) * 0.00001
        
        self.active_cells = np.array([], dtype=np.int32)

        self.pre_predicted_active_input = np.zeros((input_dim, self.history_length), dtype=np.int32)
        self.state = self.SState([],[])

    def run(self, active_input, predicted_input, learning=True):
        predicted_active_input = active_input & predicted_input
        active_overlap = self.proximal_projection.process(active_input)
        predicted_overlap = self.proximal_projection.process(predicted_active_input)
        
        total_overlap = active_overlap * self.active_overlap_weight + predicted_overlap * self.predicted_active_overlap_weight
        
        boosted_overlaps = self.boosting.process(total_overlap)
        active_columns = self.inhibition.process(boosted_overlaps)
        
        self.update_pooling(total_overlap, active_columns)
        
        # Get pooled output
        pooled_output = self.get_most_active_cells()
        
        # Compute Union SDR
        upper_overlap = self.proximal_projection_upper.process(pooled_output)
        union_SDR = self.inhibition.process(upper_overlap)
        
        if learning:
            self.proximal_projection.update(predicted_active_input, active_columns,
                                            permanence_incr=self.syn_perm_active_inc, 
                                            permanence_decr=self.syn_perm_inactive_dec)
            
            self.proximal_projection.update(pooled_output, union_SDR,
                                            permanence_incr=self.syn_perm_pred_active_inc, 
                                            permanence_decr=0.0)
            
            for i in range(self.history_length):
                if np.count_nonzero(self.pre_predicted_active_input[:,i]) == 0:
                    break
                self.proximal_projection.update(self.pre_predicted_active_input[:,i], active_columns,
                                            permanence_incr=self.syn_perm_previous_pred_active_inc, 
                                            permanence_decr=0.1)
            
        self.boosting.update(active_columns)
        
        #Save previous inputs
        self.pre_predicted_active_input = np.roll(self.pre_predicted_active_input, 1, 1)
        if self.history_length > 0:
            self.pre_predicted_active_input[:,0] = predicted_active_input
            
        self.state = self.SState(active_columns, union_SDR, overlaps=total_overlap, boosted_overlaps=boosted_overlaps)
        
        return self.state
    
    def update_pooling(self, overlaps, active_cells):
        self.pooling_activation *= (1 - self.alpha)
        
        # Sigmoid activation
        self.pooling_activation[active_cells] += overlaps[active_cells] * self.alpha
        
        return self.pooling_activation
    
    def get_most_active_cells(self):
        top_cells = self.inhibition.process(self.pooling_activation)
                
        if max(self.pooling_timer) > self.min_history:
            pooled_output = np.sort(top_cells).astype(np.int32)
        else:
            pooled_output = []
            
        return pooled_output