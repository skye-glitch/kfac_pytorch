"""Utilities for getting optimizers."""
from __future__ import annotations

import argparse
from typing import Callable

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required

import kfac
from examples.utils import create_lr_schedule
#todo new import
import math
import numpy as np


def get_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[
    optim.Optimizer,
    kfac.preconditioner.KFACPreconditioner | None,
    tuple[
        optim.lr_scheduler._LRScheduler,
        kfac.scheduler.LambdaParamScheduler | None,
    ],
]:
    """Get optimizer, preconditioner, and scheduler."""
    use_kfac = True if args.kfac_inv_update_steps > 0 else False
    optimizer = create_optimizer(args, model.parameters())
    #TODO: add new param
    lrs = create_lr_schedule(
        dist.get_world_size(),
        args.warmup_epochs,
        args.lr_decay,
        args.lars,
    )
    #print("2 optimizer, ",optimizer)
    #todo: only need the scheduler for SGD
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lrs
        ) if not args.lars else  None
    
    #print("3 optimizer, ",optimizer)
    grad_worker_fraction: kfac.enums.DistributedStrategy | float
    if args.kfac_strategy == 'comm-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.COMM_OPT
    elif args.kfac_strategy == 'mem-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.MEM_OPT
    elif args.kfac_strategy == 'hybrid-opt':
        grad_worker_fraction = args.kfac_grad_worker_fraction
    else:
        raise ValueError(
            f'Unknown KFAC Comm Method: {args.kfac_strategy}',
        )

    if use_kfac:
        preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            #todo: global_lr for lars
            lr=lambda x: optimizer.param_groups[0]['lr'] if not args.lars 
             else optimizer.global_lr,
            accumulation_steps=args.batches_per_allreduce,
            allreduce_bucket_cap_mb=25,
            colocate_factors=args.kfac_colocate_factors,
            compute_method=kfac.enums.ComputeMethod.INVERSE
            if args.kfac_inv_method
            else kfac.enums.ComputeMethod.EIGEN,
            grad_worker_fraction=grad_worker_fraction,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            skip_layers=args.kfac_skip_layers,
            #TODO: add extra param
            decay=args.decay,
        )


        def get_lambda(
            alpha: int,
            epochs: list[int] | None,
        ) -> Callable[[int], float]:
            """Create lambda function for param scheduler."""
            if epochs is None:
                _epochs = []
            else:
                _epochs = epochs

            def scale(epoch: int) -> float:
                """Compute current scale factor using epoch."""
                factor = 1.0
                for e in _epochs:
                    if epoch >= e:
                        factor *= alpha
                return factor

            return scale

        #TODO: new lambda function 
        def get_lambda_decay (
            preconditioner,
            scale: float = 0.266,
            shift: float = 40,
            avg: bool = False,
        ):
            def decay(
                epoch: int,
            ):    
                if not avg:
                    def sigmoid(x, shift, scale):
                        exp = np.exp(-scale*(x-shift))
                        return (1/(1+exp))
                    preconditioner._damping = sigmoid(epoch, shift, scale)
                else:
                    def sigmoid(x, shift, scale):
                        exp = np.exp(-scale*(x-shift))
                        return (1/(1+exp))
                    preconditioner._damping = 0.95 * preconditioner._damping + 0.05 * sigmoid(epoch, shift, scale)
                #todo: remove test code
                #print("new damping is {}".format(preconditioner._damping))
            return decay
            


        kfac_param_scheduler = kfac.scheduler.LambdaParamScheduler(
            preconditioner,
            damping_lambda=get_lambda_decay(
                preconditioner=preconditioner,
                scale=args.scale,
                shift=args.shift,
            ) 
            if args.decay else \
            get_lambda(
                args.kfac_damping_alpha,
                args.kfac_damping_decay,
            ),
            factor_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
            inv_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
            decay=args.decay,
        )
    else:
        preconditioner = None
        kfac_param_scheduler = None
    return optimizer, preconditioner, (lr_scheduler, kfac_param_scheduler)

class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_epoch=200,
                 workers=4, warmup_epochs=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        #todo: remove test code
        self.last_epoch = 0
        self.global_lr = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch,
                        workers=workers, warmup_epochs=warmup_epochs)
        super(LARS, self).__init__(params, defaults)


    def step(self, epoch: int):
        """Performs a single optimization step.
        Arguments:
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        #todo: remove test code
        #print("epochs: ",self.last_epoch, epoch)
        if self.last_epoch != epoch:
            toPrint = True
            self.last_epoch = epoch
        else:
            toPrint = False
    
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']
            warmup_epochs = group['warmup_epochs']
            workers = group['workers']
            #todo, remove test code
            i = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = p.norm()
                grad_norm = p.grad.norm()

                # Global LR warmup + 
                # computed on polynomial decay schedule
                if epoch < warmup_epochs:
                    decay = (
                        1.0 / workers * (epoch * (workers - 1) / warmup_epochs + 1)
                    )
                else:
                    decay = (1 - epoch / max_epoch) ** 2
                self.global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * self.global_lr
                #todo, remove test code
                i += 1
                # if dist.get_rank() == 0:
                #     if toPrint and actual_lr.item() > 1e-8:
                #         print("printing actual lr for epoch {}, param {}, {}".format(epoch, i, actual_lr.item()))
                    

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p + weight_decay * p.data, alpha=actual_lr.item())
                p.data.add_(-buf)
        return loss

class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(
            model_params,
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'adagrad':
        return optim.Adagrad(
            model_params,
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    elif args.optim == 'adam':
        return optim.Adam(
            model_params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'amsgrad':
        return optim.Adam(
            model_params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )
    elif args.optim == 'adabound':
        return AdaBound( 
            model_params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            final_lr=args.final_lr, 
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    elif args.optim == 'amsbound':
        return AdaBound( 
            model_params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            final_lr=args.final_lr, 
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            amsbound=True,
            eps=args.eps,
        )
    else:
        assert args.optim == 'lars'
        return LARS( #different optimizer
            model_params, 
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay, 
            eta=args.trust_coef,
            max_epoch=args.epochs, 
            workers=dist.get_world_size(),
            warmup_epochs=args.warmup_epochs,
        )


