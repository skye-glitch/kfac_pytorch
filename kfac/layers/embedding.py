import torch

from . import utils
from .base import KFACLayer
from .. import comm

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
        self.use_eigen_decomp = False
        self.hybrid_eign_decomp = True
        self.A_factor = None
        self.A_inv = None


    def _init_A_buffers(self, factor):
        """Create buffers for factor A and its inv
        For embedding layers, A is a diagonal matrix so we just store the
        diagonal to save memory.
        """
        shape = factor.shape
        self.A_factor = factor.new(shape).fill_(1)
        self.A_inv = factor.new_zeros(shape).to(self.inv_dtype)


    def _get_A_factor(self, a_inputs):
        """Compute A for Embedding layer

        Input to Embedding layer is (seq_len, batch_size) or (batch_size, seq_len)
        representing indicies into the embedding matrix of size (vocab_size, embed_size).
        The factor is represented by (1/batch_size) sum_{i} diag(one_hot(inputs[i]) ** 2)
        where inputs[i] is the input for the ith batch and the output is size [vocab_size,
        vocab_size].

        Note: If the embedding matrix has its weights tied to a decoding linear layer,
          and the layers are registerd as a shared weight KFAC layer, then self.a_inputs
          will be a mix of tensors of shape (seq_len, batch_size) and (seq_len, batch_size,
          num_tokens). I.e. the contributions from the linear layer are already one hot
          encoded so we just need to one hot encode the contributions from the embedding
          layer.

        Reference:
          https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L1107
        """
        # one hot encode all non-one-hot encoded inputs
        for i, a in enumerate(a_inputs):
            if a.size(-1) != self.module.num_embeddings:
                one_hot = torch.nn.functional.one_hot(a.long(),
                        num_classes=self.module.num_embeddings)
                a_inputs[i] = one_hot.float()
        a = utils.reshape_data(a_inputs, batch_first=self.batch_first,
                collapse_dims=True)
        assert a.size(-1) == self.module.num_embeddings
        assert len(a.shape) == 2  # shape should be (batch, vocab_size) where batch dim
                                  # has size batch_size * seq_len
        a = a ** 2
        return  torch.mean(a, dim=0)

    # Implement update_A_factor here...
    #   we should not use base.update_A_factor
    #   as the state['A'] definition there
    def update_A_factor(self, alpha=0.95):
        if len(self.a_inputs) == 0:
            return
        self.a_inputs = [x.to(self.factor_dtype) for x in self.a_inputs]
        A_new = self._get_A_factor(self.a_inputs)
        # print("A sizes", len(self.a_inputs), self.a_inputs[0].shape, self.module.num_embeddings, A_new.shape)
        del self.a_inputs[:]  # clear accumulated inputs
        if self.A_factor is None:
            self._init_A_buffers(A_new)
            self.state['A_shape'] = A_new.shape
        utils.update_running_avg(A_new, self.A_factor, alpha=alpha)


    def compute_A_inv(self, damping=0.001, ignore_rank=False):
        if self.compute_A_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if self.keep_inv_copy is None:
            raise ValueError('Grad workers have not been assigned to layer yet.')
        if self.A_factor is None:
            raise RuntimeError('update_A_factor() must be called at least '
                               'once before calling compute_A_inv().')

        if ignore_rank or comm.backend.rank() == self.compute_A_inv_rank:
            self.A_inv.copy_(
                utils.get_elementwise_inverse(self.A_factor, damping) ## no need to covert; .copy_() does it for us; .to(self.inv_dtype)
            )

    def _get_G_factor(self, g_inputs):
        # g_inputs size: [batch_size, seq_len, num_embeddings]
        g = utils.reshape_data(g_inputs, batch_first=self.batch_first,
                collapse_dims=True)
        # g size: [batch_size * seq_len, num_embeddings]
        assert len(g.shape) == 2  # shape should be (batch, num_embeddings) where batch dim
                                # has size batch_size * seq_len
        # print("G shape", g_inputs.size(), g.shape, self.module.num_embeddings)
        return utils.get_cov(g)

    # def update_G_factor
    #   reuse base.update_G_factor

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
            # let's be safe first to create buffer for all IRs
            if 'QG' not in self.state:
                self.state['QG'] = torch.empty_like(
                        self.state['G'], dtype=self.inv_dtype)
            if 'dG' not in self.state:
                self.state['dG'] = self.state['G'].new_empty(
                            self.state['G'].shape[0], dtype=self.inv_dtype)
            if 'G_inv' not in self.state:
                self.state['G_inv'] = torch.empty_like(
                          self.state['G'], dtype=self.inv_dtype)

        def _eigen_decomp(factor, damping=0.001, symm=True):
            Q, d = utils.get_eigendecomp(factor.to(torch.float32), concat=False,
                                         symmetric=symm)
            return Q.to(self.inv_dtype), d.to(self.inv_dtype)

        if ignore_rank or comm.backend.rank() == self.compute_G_inv_rank:
            results = _eigen_decomp(self.state['G'], damping, self.factors_are_symmetric)
            self.state['QG'] = results[0]
            self.state['dG'] = results[1]
            QG = results[0]
            dG = results[1]
            Ginv = QG @ torch.diag(utils.get_elementwise_inverse(dG)) @ QG.t()
            self.state['G_inv'] = (Ginv + Ginv.t()) / 2 # there is some tiny pricision errors in fp32


    def _get_precondition_gradient_hybrid(self):
        """Compute preconditioned gradient using specified inverse method"""
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Reconstruct inv if it was flattened for communication
            # No need for A_inv in embedding layers, always as a vector
            if len(self.state['G_inv'].shape) == 1:
                rows, cols = self.state['G'].shape
                self.state['G_inv'] = utils.fill_triu(
                        [rows, cols], self.state['G_inv'])

        """
        Note: For embedding layers, A is a diagonal matrix stored as a 1-D
        tensors of the diagonal and the gradient is (input_size, output_size).
        The KFAC update expects the gradient to be (output_size, input_size)
        so we use this update:
            precon_grad = (G_inv @ grad.t()) * A_inv
        instead of the traditional:
            precon_grad = G_inv.t() @ grad @ A_inv
        where @ is torch.matmul() and * is torch.mv()/
        """
        grad = self.get_gradient().to(self.inv_dtype)
        # print("printing gradient type and shape", grad.dtype, grad.shape, self.A_inv.shape, self.state['G_inv'].shape)
        res1 = grad @ self.state['G_inv']
        res2 = self.A_inv[:, None] * res1 
        return res2.to(torch.float32)
        # return (self.A_inv[:, None] * (grad @ self.state['G_inv'])).to(torch.float32)


    def _get_precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        raise NotImplementedError('Use hybrid (inv for A, eigen for G)' +
                                                'method for embedding layers')

    def _get_precondition_gradient_inv(self):
        raise NotImplementedError('Use hybrid (inv for A, eigen for G)' +
                                                'method for embedding layers')

    # Communication rewrite
    def allreduce_factors(self):
        """Allreduce self.A_factor and self.state['G'] factors

        Returns:
          list of async work handles
        """
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle for G
            self.state['G_flat'] = utils.get_triu(self.state['G'])
            return [comm.backend.allreduce(self.A_factor),
                    comm.backend.allreduce(self.state['G_flat'])]
        return [comm.backend.allreduce(self.A_factor),
                comm.backend.allreduce(self.state['G'])]


    def broadcast_inverses(self):
        """Broadcast self.A_inv and self.state['G_inv'] inverses

        Note: all ranks enter this function but some ranks may not be in the
          broadcast group for the inverses. comm.backend.broadcast() will be a
          no-op if a group is provided in rank is not in the group.

        Returns:
          list of async work handles
        """
        if not self.keep_inv_copy:
            return []

        ops = [
               comm.backend.broadcast(self.state['QG'], 
                       src=self.compute_G_inv_rank,
                       group=self.broadcast_G_inv_group)]

        ops.append(comm.backend.broadcast(self.state['dG'], 
                    src=self.compute_G_inv_rank, 
                    group=self.broadcast_G_inv_group))

        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle
            # self.state['A_inv'] = utils.get_triu(self.state['A_inv'])
            self.state['G_inv'] = utils.get_triu(self.state['G_inv'])
        ops.append(comm.backend.broadcast(self.A_inv, 
                        src=self.compute_A_inv_rank,
                        group=self.broadcast_A_inv_group))
        
        ops.append(comm.backend.broadcast(self.state['G_inv'], 
                        src=self.compute_G_inv_rank,
                        group=self.broadcast_G_inv_group))
        return ops

