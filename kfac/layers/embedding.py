import torch
import math

from . import utils
from .base import KFACLayer
from .. import comm

#add
import logging
'''
import eighMG
'''
import numpy as np

class EmbeddingLayer(KFACLayer):
    """
    Note:
      Defaults to batch_first=False

    Note:
      If the embedding weights are tied with the Linear "decoding" weights, then
      the forward and backward pass contributions from the Linear module will not be
      correctly incorporated.
    """
    def __init__(self, *args, **kwargs):
        super(EmbeddingLayer, self).__init__(*args, **kwargs)
        # TODO(gpauloski): update embedding class to support recent KFAC changes
        #raise ValueError('Embedding layer does not currently work')
        self.has_bias = False
        self.use_eigen_decomp = True
        #self.hybrid_eign_decomp = True
        self.A_factor = None
        self.A_inv = None


    # def _init_A_buffers(self, factor, damping=0.001):
    #     """Create buffers for factor A and its inv
    #     For embedding layers, A is a diagonal matrix so we just store the
    #     diagonal to save memory.
    #     """
    #     shape = factor.shape
    #     self.A_factor = factor.new(shape).fill_(1)
    #     self.A_inv = factor.new(shape).fill_(damping).to(self.inv_dtype) # instead of zero, we init it to damping


    # def _compute_factor_inverse_mg(self, factor, damping=0.001):
    # #Computes inverse/eigendecomp of factor and saves result to inverse
    #     Q, d = eighMG.syevdMG(factor.to(torch.float32), 8)
    #     return Q.to(self.inv_dtype), d.to(self.inv_dtype)

    def _get_A_factor(self, a_inputs):
        # a: batch_size * in_dim
        for i, a in enumerate(a_inputs):
            if a.size(-1) != self.module.num_embeddings:
                one_hot = torch.nn.functional.one_hot(a.long(),
                        num_classes=self.module.num_embeddings)
                a_inputs[i] = one_hot.float()
        a = utils.reshape_data(a_inputs, batch_first=self.batch_first,
                collapse_dims=True, toflt=True)
        #print("a inputs size {}, reshape size {}".format(a_inputs[0].shape, a.shape))
        return utils.get_cov(a)

    def update_A_factor(self, alpha=0.95):
        """Compute factor A and add to running averages"""
        if len(self.a_inputs) == 0:
            return
        # for i in self.a_inputs:
        #     _t = i.cpu()
        #     if not np.isfinite(_t).all():
        #         logging.critical("find inf in a_inputs 1 {}".format(self.module))
        #     break
        self.a_inputs = [x.to(self.factor_dtype) for x in self.a_inputs]
        # for i in self.a_inputs:
        #     _t = i.cpu()
        #     if not np.isfinite(_t).all():
        #         logging.critical("find inf in a_inputs 2 {}".format(self.module))
        #     break
        A_new = self._get_A_factor(self.a_inputs)
        # TODO check A_new inf 
        # for i in A_new:
        #     _t = i.cpu()
        #     if not np.isfinite(_t).all():
        #         logging.critical("find inf in A_new {}".format(self.module))
        #     break
        del self.a_inputs[:]  # clear accumulated inputs
        if self.state['A'] is None:
            self.state['A'] = torch.diag(A_new.new(A_new.shape[0]).fill_(1))
            self.state['A_shape'] = A_new.shape
        utils.update_running_avg(A_new, self.state['A'], alpha=alpha)
        # TODO: check self.state['A']
        # for i in self.state['A']:
        #     _t = i.cpu()
        #     if not np.isfinite(_t).all():
        #         logging.critical("find inf in self.state['A'] {}".format(self.module))
        #     break


    def compute_A_inv(self, damping=0.001, ignore_rank=False):
        """Compute A inverse on assigned rank

        Note: 
          - all ranks will enter this function but only the ranks assigned
            to this layer will continue to actually compute the inverses.
            All other ranks will simply zero out their inverse buffers for.
            This is done so we can sum the inverses across all ranks to 
            communicate the results of locally computed inverses.
          - tensors for storing the inverse will be initialized based on the
            shape of the factor if the inv is None. This means that
            self.update_A_factor() must be called at least once before this
            function.

        Args:
          damping (float, optional): damping value to condition inverse 
             (default: 0.001)
          ignore_rank (bool, optional): ignore assigned rank and compute
             inverse (default: False)
        """
        if self.compute_A_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if self.keep_inv_copy is None:
            raise ValueError('Grad workers have not been assigned to layer yet.')
        if self.state['A'] is None:
            raise RuntimeError('update_A_factor() must be called at least '
                               'once before calling compute_A_inv().')
        
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Reconstruct factor if it was flattened for communication
            if 'A_flat' in self.state:
                self.state['A'] = utils.fill_triu(
                        self.state['A'].shape, self.state['A_flat'])
                del self.state['A_flat']

        # Init inv buffer for ranks that will receive the inverse
        if self.keep_inv_copy:
            if self.use_eigen_decomp and 'QA' not in self.state:
                self.state['QA'] = torch.empty_like(
                        self.state['A'], dtype=self.inv_dtype)
                if self.prediv_eigenvalues and 'dGdA' not in self.state:
                    self.state['dGdA'] = self.state['A'].new_empty(
                            (self.state['G'].shape[0], self.state['A'].shape[0]),
                            dtype=self.inv_dtype)
                elif not self.prediv_eigenvalues and 'dA' not in self.state:
                    self.state['dA'] = self.state['A'].new_empty(
                            self.state['A'].shape[0], dtype=self.inv_dtype)
            elif not self.use_eigen_decomp and 'A_inv' not in self.state:
                self.state['A_inv'] = torch.empty_like(
                        self.state['A'], dtype=self.inv_dtype)

        if ignore_rank or comm.backend.rank() == self.compute_A_inv_rank:
            results = self._compute_factor_inverse(self.state['A'], damping)

            """
            version1, add barrier? 
            nbGPU = 8 if self.state['A'].shape[0] > 16000 else 1
            if nbGPU > 1:
                if comm.backend.rank() != self.compute_A_inv_rank: 
                    torch.distributed.barrier() 
                else:
                    results = self._compute_factor_inverse_mg(self.state['A'], damping)
                    torch.distributed.barrier() 
            else:
                if ignore_rank or comm.backend.rank() == self.compute_A_inv_rank:
                    results = self._compute_factor_inverse(self.state['A'], damping)
            """

            if isinstance(results, tuple):
                self.state['QA'] = results[0]
                self.state['dA'] = results[1]
                # if not torch.isfinite(self.state['QA']).all():
                #     logging.critical("find inf in self.state['QA']")
                # if not torch.isfinite(self.state['dA']).all():
                #     logging.critical("find inf in self.state['dA']")
            else:
                self.state['A_inv'] = results
                # if not torch.isfinite(self.state['A_inv']).all():
                #     logging.critical("find inf in self.state['A_inv']")

    def _get_G_factor(self, g_outputs): 
        # g: batch_size * out_dim
        g = utils.reshape_data(g_outputs, batch_first=self.batch_first,
                collapse_dims=True)
        #print("g inputs size {}, reshape size {}".format(g_outputs[0].shape, g.shape))
        return utils.get_cov(g)

  

    def compute_G_inv(self, damping=0.001, ignore_rank=False):
        """Compute G inverse on specified ranks

        See `compute_A_inv` for more info`
        """
        if self.compute_G_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if self.keep_inv_copy is None:
            raise ValueError('Grad workers have not been assigned to layer yet.')
        if self.state['G'] is None:
            raise RuntimeError('update_G_factor() must be called at least '
                               'once before calling compute_G_inv().')
        
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Reconstruct factor if it was flattened for communication
            if 'G_flat' in self.state:
                self.state['G'] = utils.fill_triu(
                        self.state['G'].shape, self.state['G_flat'])
                del self.state['G_flat']

        # Init inv buffer for ranks that will receive the inverse
        if self.keep_inv_copy:
            if self.use_eigen_decomp and 'QG' not in self.state:
                self.state['QG'] = torch.empty_like(
                        self.state['G'], dtype=self.inv_dtype)
                if self.prediv_eigenvalues and 'dGdA' not in self.state:
                    self.state['dGdA'] = self.state['A'].new_empty(
                            (self.state['G'].shape[0], self.state['A'].shape[0]),
                            dtype=self.inv_dtype)
                elif not self.prediv_eigenvalues and 'dG' not in self.state:
                    self.state['dG'] = self.state['G'].new_empty(
                            self.state['G'].shape[0], dtype=self.inv_dtype)
            elif not self.use_eigen_decomp and 'G_inv' not in self.state:
                self.state['G_inv'] = torch.empty_like(
                        self.state['G'], dtype=self.inv_dtype)

        if ignore_rank or comm.backend.rank() == self.compute_G_inv_rank:
            results = self._compute_factor_inverse(self.state['G'], damping)

            if isinstance(results, tuple):
                self.state['QG'] = results[0]
                self.state['dG'] = results[1]

                # if not torch.isfinite(self.state['QG']).all():
                #     logging.critical("find inf in self.state['QG']")
                # if not torch.isfinite(self.state['dG']).all():
                #     logging.critical("find inf in self.state['dG']")

                if self.prediv_eigenvalues:
                    if 'dA' not in self.state:
                        raise ValueError('compute_A_inv must be called before '
                                         'compute_G_inv if prediv_eigenvalues '
                                         'is True.')
                    self.state['dGdA'] = 1 / (self.state['dG'].unsqueeze(1) *
                            self.state['dA'].unsqueeze(0) + damping)
                    # if not torch.isfinite(self.state['dGdA']).all():
                    #     logging.critical("find inf in self.state['dGdA']")
            else:
                self.state['G_inv'] = results
                # if not torch.isfinite(self.state['G_inv']).all():
                #         logging.critical("find inf in self.state['G_inv']")

            

      

    def _get_precondition_gradient_eigen(self, damping=0.001):
        """Compute preconditioned gradient for eigendecomp method"""
        QA = self.state['QA']
        QG = self.state['QG']
        grad = self.get_gradient().to(self.inv_dtype).t()
        #print("QG shape {}, grad shape {}, QA shape {}".format(QG.shape, grad.shape, QA.shape))
        v1 = QG.t() @ grad @ QA
        if self.prediv_eigenvalues:
            v2 = v1 * self.state['dGdA']
        else:
            v2 = v1 / (self.state['dG'].unsqueeze(1) * 
                       self.state['dA'].unsqueeze(0) + damping)
        return (QG @ v2 @ QA.t()).to(torch.float32)

    def _get_precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient().to(self.inv_dtype).t()
        return (self.state['G_inv'] @ grad @ self.state['A_inv']).to(torch.float32)
   
   
    # Communication rewrite
    def allreduce_factors(self):
        """Allreduce A and G factors

        Returns:
          list of async work handles
        """
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle
            self.state['A_flat'] = utils.get_triu(self.state['A'])
            self.state['G_flat'] = utils.get_triu(self.state['G'])
            return [comm.backend.allreduce(self.state['A_flat']),
                    comm.backend.allreduce(self.state['G_flat'])]
        return [comm.backend.allreduce(self.state['A']),
                comm.backend.allreduce(self.state['G'])]


    def broadcast_inverses(self):
        """Broadcast A and G inverses

        Note: all ranks enter this function but some ranks may not be in the
          broadcast group for the inverses. comm.backend.broadcast() will be a
          no-op if a group is provided in rank is not in the group.

        Returns:
          list of async work handles
        """
        if not self.keep_inv_copy:
            return []

        if self.use_eigen_decomp:
            ops = [comm.backend.broadcast(self.state['QA'],
                           src=self.compute_A_inv_rank,
                           group=self.broadcast_A_inv_group),
                   comm.backend.broadcast(self.state['QG'], 
                           src=self.compute_G_inv_rank,
                           group=self.broadcast_G_inv_group)]
            if self.prediv_eigenvalues:
                ops.append(comm.backend.broadcast(self.state['dGdA'], 
                        src=self.compute_A_inv_rank, 
                        group=self.broadcast_A_inv_group))
            else:
                ops.append(comm.backend.broadcast(self.state['dA'], 
                        src=self.compute_A_inv_rank, 
                        group=self.broadcast_A_inv_group))
                ops.append(comm.backend.broadcast(self.state['dG'], 
                        src=self.compute_G_inv_rank, 
                        group=self.broadcast_G_inv_group))
            return ops

        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle
            self.state['A_inv'] = utils.get_triu(self.state['A_inv'])
            self.state['G_inv'] = utils.get_triu(self.state['G_inv'])
        return [comm.backend.broadcast(self.state['A_inv'], 
                        src=self.compute_A_inv_rank,
                        group=self.broadcast_A_inv_group),
                comm.backend.broadcast(self.state['G_inv'], 
                        src=self.compute_G_inv_rank,
                        group=self.broadcast_G_inv_group)]

   