import math
from itertools import count

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .physics import clean_force, local_energy
from .sampling import samples_from
from .utils import NULL_DEBUG, normalize_mean, state_dict_copy, weighted_mean_var


def loss_local_energy(Es_loc, psis, ws, E_ref=None, p=1):
    assert psis.grad_fn is None
    E0 = E_ref if E_ref is not None else (ws * Es_loc).mean()
    return (ws * (Es_loc - E0).abs() ** p).mean()


class LossWeightedLogProb:
    def __call__(self, Es_loc, psis, ws):
        assert Es_loc.grad_fn is None
        self.weights = ws * (Es_loc - (ws * Es_loc).mean())
        return 2 * (self.weights * torch.log(psis.abs())).mean()


def loss_least_squares(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def fit_wfnet_multi(wfnet, loss_funcs, opts, gen_factory, gen_kwargs, writers):
    for loss_func, opt, kwargs, writer in zip(loss_funcs, opts, gen_kwargs, writers):
        with writer:
            fit_wfnet(wfnet, loss_func, opt, gen_factory(**kwargs), writer=writer)


def outlier_mask(x, p, q, dim=None):
    x = x.detach()
    dim = dim if dim is not None else -1
    n = x.shape[dim]
    lb = x.kthvalue(int(p * n), dim=dim).values
    ub = x.kthvalue(int((1 - p) * n), dim=dim).values
    return (
        (x - (lb + ub).unsqueeze(dim) / 2).abs() > q * (ub - lb).unsqueeze(dim),
        (lb, ub),
    )


def log_clipped_outliers(x, q):
    x = x.detach()
    median = x.median()
    x = x - median
    a = q * x.abs().mean()
    x = torch.where(
        x.abs() <= a, x, x.sign() * a * (1 + torch.log((1 + (x.abs() / a) ** 2) / 2))
    )
    return median + x


class NanLoss(Exception):
    pass


def fit_wfnet(
    wfnet,
    loss_func,
    opt,
    sampler,
    steps,
    require_energy_gradient=False,
    require_psi_gradient=True,
    clip_grad=None,
    writer=None,
    start=0,
    debug=NULL_DEBUG,
    skip_outliers=False,
    clip_outliers=False,
    p=0.01,
    q=5,
    subbatch_size=None,
    clean_tau=None,
):
    assert not (skip_outliers and clip_outliers)
    for step, (rs, psi0s) in zip(steps, sampler):
        d = debug[step]
        d['psi0s'], d['rs'], d['state_dict'] = psi0s, rs, state_dict_copy(wfnet)
        subbatch_size = subbatch_size or len(rs)
        subbatches = []
        for rs, psi0s in DataLoader(TensorDataset(rs, psi0s), batch_size=subbatch_size):
            Es_loc, psis, forces = local_energy(
                rs,
                wfnet,
                create_graph=require_energy_gradient,
                keep_graph=require_psi_gradient,
                return_grad=True,
            )
            ws = (psis.detach() / psi0s) ** 2
            if clean_tau is not None:
                forces = forces / psis.detach()[:, None, None]
                forces_clean = clean_force(forces, rs, wfnet.mol, tau=clean_tau)
                forces, forces_clean = (
                    x.flatten(start_dim=-2).norm(dim=-1) for x in (forces, forces_clean)
                )
                force_ws = forces_clean / forces
            else:
                force_ws = torch.ones_like(ws)
            total_ws = ws * force_ws
            if skip_outliers:
                outliers = outlier_mask(Es_loc, p, q)[0]
                Es_loc_loss, psis, total_ws = (
                    Es_loc[~outliers],
                    psis[~outliers],
                    total_ws[~outliers],
                )
            elif clip_outliers:
                Es_loc_loss = log_clipped_outliers(Es_loc, q)
            else:
                Es_loc_loss = Es_loc
            loss = loss_func(Es_loc_loss, psis, normalize_mean(total_ws))
            loss.backward()
            subbatches.append(
                (
                    loss.detach().view(1),
                    Es_loc.detach(),
                    Es_loc_loss.detach(),
                    psis.detach(),
                    ws,
                    force_ws,
                    total_ws,
                )
            )
        loss, Es_loc, Es_loc_loss, psis, ws, forces_ws, total_ws = (
            torch.cat(xs) for xs in zip(*subbatches)
        )
        loss = d['loss'] = loss.sum()
        if torch.isnan(loss).any():
            raise NanLoss()
        d['Es_loc'], d['psis'] = Es_loc, psis
        if clip_grad:
            clip_grad_norm_(wfnet.parameters(), clip_grad)
        if writer:
            E_loc_mean, E_loc_var = weighted_mean_var(Es_loc, ws)
            writer.add_scalar('E_loc/mean', E_loc_mean, step)
            writer.add_scalar('E_loc/var', E_loc_var, step)
            E_loc_loss_mean, E_loc_loss_var = weighted_mean_var(Es_loc_loss, total_ws)
            writer.add_scalar('E_loc_loss/mean', E_loc_loss_mean, step)
            writer.add_scalar('E_loc_loss/var', E_loc_loss_var, step)
            writer.add_scalar('psi/sq_mean', (psis ** 2).mean(), step)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('weights/mean', ws.mean(), step)
            writer.add_scalar('weights/median', ws.median(), step)
            writer.add_scalar('weights/var', ws.var(), step)
            writer.add_scalar('force_weights/min', force_ws.min(), step)
            writer.add_scalar('force_weights/max', force_ws.max(), step)
            grads = torch.cat(
                [p.grad.flatten() for p in wfnet.parameters() if p.grad is not None]
            )
            writer.add_scalar('grad/norm', grads.norm(), step)
            for label, value in wfnet.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
            lr = opt.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('misc/learning_rate', lr, step)
            writer.add_scalar('misc/batch_size', len(Es_loc), step)
        opt.step()
        opt.zero_grad()
        yield step


def batched_sampler(
    sampler,
    *,
    batch_size,
    sample_every,
    n_discard=0,
    n_decorrelate=0,
    range_sampling=range,
):
    if isinstance(batch_size, int):
        batch_size = [(None, batch_size)]
    for n_epochs, bs in batch_size:
        for _ in range(n_epochs) if n_epochs else count():
            n_sampling_steps = (
                math.ceil(sample_every * bs / len(sampler)) * (1 + n_decorrelate)
                + n_discard
            )
            sampler.restart()
            rs, psis, _ = samples_from(
                sampler,
                range_sampling(n_sampling_steps),
                n_discard=n_discard,
                n_decorrelate=n_decorrelate,
            )
            samples_ds = TensorDataset(rs.flatten(end_dim=1), psis.flatten(end_dim=1))
            rs_dl = DataLoader(samples_ds, batch_size=bs, shuffle=True, drop_last=True)
            yield from rs_dl


def simple_sampler(sampler, *, n_discard=0, n_decorrelate=0):
    while True:
        rs, psis, _ = samples_from(
            sampler, range(1), n_discard=n_discard, n_decorrelate=n_decorrelate
        )
        yield rs.flatten(end_dim=1), psis.flatten(end_dim=1)


def fit_wfnet_supervised(
    fit_net,
    true_net,
    loss_func,
    opt,
    sample_gen,
    correlated_sampling=True,
    acc_grad=1,
    writer=None,
    start=0,
    debug=NULL_DEBUG,
    scheduler=None,
    epoch_size=100,
):
    for step, (rs, psi0s) in enumerate(sample_gen, start=start):
        d = debug[step]
        d['psi0s'], d['rs'] = psi0s, rs
        psis_fit = fit_net(rs)
        psis_true = true_net(rs)
        loss = loss_func(psis_fit, psis_true)
        if writer:
            writer.add_scalar('loss', loss, step)
            for label, value in fit_net.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
        loss.backward()
        if (step + 1) % acc_grad == 0:
            opt.step()
            opt.zero_grad()
        d['state_dict'] = state_dict_copy(fit_net)
        if scheduler and (step + 1) % epoch_size == 0:
            scheduler.step()