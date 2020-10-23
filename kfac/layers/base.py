import math
import torch
import warnings

from . import utils
from .. import comm

class KFACLayer(object):
    def __init__(self,
                 module,
                 accumulate_data=True,
                 batch_first=True,
                 grad_scaler=None,
                 keep_inv_copy=False,
                 use_eigen_decomp=True,
                 use_half_precision=False,
                 compute_A_inv_rank=None,
                 compute_G_inv_rank=None,
                 compute_grad_ranks=None):
        self.module = module
        self.accumulate_data = accumulate_data
        self.batch_first = batch_first
        self.grad_scaler = grad_scaler
        self.keep_inv_copy = keep_inv_copy
        self.use_eigen_decomp=use_eigen_decomp
        self.compute_A_inv_rank = compute_A_inv_rank
        self.compute_G_inv_rank = compute_G_inv_rank
        self.compute_grad_ranks = compute_grad_ranks
        self.eps = 1e-10

        # Should be overridden by implementing class
        self.has_bias = False
        self.factors_are_symmetric = True

        self.a_inputs  = []  # inputs accumulated from module hooks
        self.g_outputs = []  # outputs accumulated from module hooks
        self.A_factor  = None
        self.G_factor  = None
        self.A_inv     = None
        self.G_inv     = None
        self.preconditioned_gradient = None

    def __repr__(self):
        return 'KFAC {}({})'.format(self.__class__.__name__, repr(self.module))

    def state_dict(self, include_inverses=False):
        """Returns the state of the KFACLayer as a dictionary.

        Used by kfac.KFAC for state saving/loading. Note by default only the
        factors are saved because the inverses can be recomputed from the
        factors, however, the `include_inverses` flag can override this. If
        `keep_inv_copy=False` and `compute_A_inv_rank != compute_G_inv_rank != get_rank()` then
        the inverses may be `None` because this worker is not responsible for
        this layer.
        """
        state = {'A_factor': self.A_factor, 'G_factor': self.G_factor}
        if include_inverses:
            state['A_inv'] = self.A_inv
            state['G_inv'] = self.G_inv
        return state

    def load_state_dict(self, state_dict):
        """Loads the KFACLayer state."""
        device = next(self.module.parameters()).device
        try:
            self._init_A_buffers(state_dict['A_factor'].to(device))
            self._init_G_buffers(state_dict['G_factor'].to(device))
            if 'A_inv' in state_dict:
                self.A_inv = state_dict['A_inv']
            if 'G_inv' in state_dict:
                self.G_inv = state_dict['G_inv'] 
        except KeyError: 
            raise KeyError('KFACLayer state_dict must contain keys: '
                           '["A_factor", "G_factor"].')

    def assign_workers(self, compute_A_inv_rank, compute_G_inv_rank,
                compute_grad_ranks):
        """Assign ranks to this layer

        Args:
          compute_A_inv_rank (int): rank to compute A inverse on
          compute_G_inv_rank (int): rank to compute G inverse on
          compute_grad_ranks (list(int)): ranks that should compute
                  preconditioned gradient
        """
        self.compute_A_inv_rank = compute_A_inv_rank
        self.compute_G_inv_rank = compute_G_inv_rank
        self.compute_grad_ranks = compute_grad_ranks

    def allreduce_factors(self):
        """Allreduce A and G factors

        Returns:
          list of async work handles
        """
        return [comm.backend.allreduce(self.A_factor),
                comm.backend.allreduce(self.G_factor)]

    def broadcast_inverses(self):
        """Broadcast A and G inverses

        Returns:
          list of async work handles
        """
        return [comm.backend.broadcast(self.A_inv, self.compute_A_inv_rank),
                comm.backend.broadcast(self.G_inv, self.compute_G_inv_rank)]

    def broadcast_gradient(self):
        """Broadcast preconditioned gradient

        Returns:
          list of async work handles
        """
        # If the preconditioned gradient is None, initialize it so
        # we can correctly perform the broadcast op between ranks
        if self.preconditioned_gradient is None:
            w = self._get_weight_grad()
            self.preconditioned_gradient = [w.new_zeros(w.shape)]
            if self.has_bias:
                b = self._get_bias_grad()
                self.preconditioned_gradient.append(b.new_zeros(b.shape))

        self.preconditioned_gradient = [t.contiguous() for t in
                self.preconditioned_gradient]
        
        return [comm.backend.broadcast(tensor, self.compute_A_inv_rank)
                for tensor in self.preconditioned_gradient]

    def compute_A_inv(self, damping=0.001, ignore_rank=False):
        """Compute A inverse on assigned rank

        Note: all ranks will enter this function but only the ranks assigned
        to this layer will continue to actually compute the inverses.
        All other ranks will simply zero out their inverse buffers for. This 
        is done so we can sum the inverses across all ranks to communicate the
        results of locally computed inverses.

        Args:
          damping (float, optional): damping value to condition inverse 
             (default: 0.001)
          ignore_rank (bool, optional): ignore assigned rank and compute
             inverse (default: False)
        """
        if self.compute_A_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if ignore_rank or comm.backend.rank() == self.compute_A_inv_rank:
            self.A_inv = self._compute_factor_inverse(
                    self.A_factor.float(), damping)

    def compute_G_inv(self, damping=0.001, ignore_rank=False):
        """Compute G inverse on specified ranks

        See `compute_A_inv` for more info`
        """
        if self.compute_G_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if ignore_rank or comm.backend.rank() == self.compute_G_inv_rank:
            self.G_inv = self._compute_factor_inverse(
                    self.G_factor.float(), damping)

    def get_gradient(self):
        """Get formated gradients (weight and bias) of module

        Returns:
          gradient of shape [ouput_dim, input_dim]. If bias != None, concats bias
        """
        g = self._get_weight_grad()
        if self.has_bias:
            g = torch.cat([g, self._get_bias_grad().data.view(-1, 1)], 1)
        return g

    def compute_preconditioned_gradient(self, damping=0.001):
        """Compute precondition gradient of each weight in module
        
        Preconditioned gradients can be applied to the actual gradients with 
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
          damping (float, optional): damping to use if preconditioning using
              the eigendecomposition method. (default: 0.001)
        """
        if self.compute_grad_ranks is None:
            raise ValueError('Gradient preconditioning workers have not been '
                             'assigned yet. Have you called assign_workers() '
                             'yet?')
        if comm.backend.rank() not in self.compute_grad_ranks:
            return

        # Compute preconditioned gradient using specified inverse method
        if self.use_eigen_decomp:
            grad = self._get_precondition_gradient_eigen(damping)
        else:
            grad = self._get_precondition_gradient_inv()
        # Reshape appropriately
        if self.has_bias:
            grad = [grad[:, :-1], grad[:, -1:]]
            grad[0] = grad[0].view(self._get_weight_grad().data.size())
            grad[1] = grad[1].view(self._get_bias_grad().data.size())
        else:
            grad = [grad.view(self._get_weight_grad().data.size())]
        self.preconditioned_gradient = grad

    def save_inputs(self, input):
        """Save inputs locally"""
        if self.accumulate_data:
            self.a_inputs.append(input[0].data)
        else:
            self.a_inputs = [input[0].data]

    def save_grad_outputs(self, grad_output):
        """Save grad w.r.t outputs locally"""
        g = grad_output[0].data
        if self.grad_scaler is not None:
            g = (g, self.grad_scaler.get_scale())
        if self.accumulate_data:
            self.g_outputs.append(g)
        else:
            self.g_outputs = [g]

    def update_A_factor(self, alpha=0.95):
        """Compute factor A and add to running averages"""
        A_new = self._get_A_factor(self.a_inputs)
        del self.a_inputs[:]  # clear accumulated inputs
        if self.A_factor is None:
            self._init_A_buffers(A_new)
        utils.update_running_avg(A_new, self.A_factor, alpha=alpha)

    def update_G_factor(self, alpha=0.95):
        """Compute factor G and add to running averages"""
        # If half precision training: unscale accumulated gradients and discard
        # any with inf/NaN because of round-off errors. We need to unscale
        # to correctly compute G.
        if self.grad_scaler is not None:
            self.g_outputs = [g / scale for g, scale in self.g_outputs]
            length = len(self.g_outputs)
            self.g_outputs = [g for g in self.g_outputs 
                    if not (torch.isinf(g).any() or torch.isnan(g).any())]
            if len(self.g_outputs) != length:
                warnings.warn('Some gradients were discarded when computing '
                              'G because they were unable to be unscaled. '
                              'Note this can degrade KFAC performance if too '
                              'many gradients are discarded.')

        G_new = self._get_G_factor(self.g_outputs)
        del self.g_outputs[:]  # clear accumulated outputs
        if self.G_factor is None:
            self._init_G_buffers(G_new)
        utils.update_running_avg(G_new, self.G_factor, alpha=alpha)

    def update_gradient(self, scale=None):
        """Updates gradients of module with computed precondition gradients"""
        if self.preconditioned_gradient is None:
            raise RuntimeError('self.compute_preconditioned_gradient() should'
                    ' be called before update_gradient()')
        if scale is not None:
            v = [scale * x for x in self.preconditioned_gradient]
        else:
            v = self.preconditioned_gradient
        self._get_weight_grad().data = v[0].data
        if self.has_bias:
            self._get_bias_grad().data = v[1].data

    def _compute_factor_inverse(self, factor, damping=0.001):
        """Computes inverse/eigendecomp of factor and saves result to inverse"""
        if self.use_eigen_decomp:
            return utils.get_eigendecomp(factor, symmetric=self.factors_are_symmetric)
        else:
            return utils.get_inverse(factor, damping=damping, 
                        symmetric=self.factors_are_symmetric)
 
    def _get_A_factor(self, a_inputs):
        """Compute A factor. Returns A."""
        raise NotImplementedError

    def _get_G_factor(self, g_inputs):
        """Compute G factor. Returns G."""
        raise NotImplementedError

    def _get_bias_grad(self):
        """Get bias.grad tensor of module"""
        return self.module.bias.grad

    def _get_weight_grad(self):
        """Get weight.grad tensor of module"""
        return self.module.weight.grad
    
    def _get_precondition_gradient_eigen(self, damping=0.001):
        """Compute preconditioned gradient for eigendecomp method"""
        grad = self.get_gradient()
        QA, QG = self.A_inv[:,:-1], self.G_inv[:,:-1]
        dA, dG = self.A_inv[:,-1], self.G_inv[:,-1]
        v1 = QG.t() @ grad @ QA
        v2 = v1 / (dG.unsqueeze(1) * dA.unsqueeze(0) + damping)
        return QG @ v2 @ QA.t()

    def _get_precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient() 
        return self.G_inv @ grad @ self.A_inv

    def _init_A_buffers(self, factor):
        """Create buffers for factor A and its inverse"""
        assert self.A_factor is None, ('A buffers have already been '
                'initialized. Was _init_A_buffers() called more than once?')
        self.A_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.keep_inv_copy:
            if self.use_eigen_decomp:
                # add one axtra column for eigenvalues
                shape = (factor.shape[0], factor.shape[0] + 1) 
            else:
                shape = factor.shape
            self.A_inv = factor.new_zeros(shape).to(torch.float32)

    def _init_G_buffers(self, factor):
        """Create buffers for factor G and its inverse"""
        assert self.G_factor is None, ('G buffers have already been '
                'initialized. Was _init_G_buffers() called more than once?')
        self.G_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.keep_inv_copy:
            if self.use_eigen_decomp:
                # add one axtra column for eigenvalues
                shape = (factor.shape[0], factor.shape[0] + 1)
            else:
                shape = factor.shape
            self.G_inv = factor.new_zeros(shape).to(torch.float32)

