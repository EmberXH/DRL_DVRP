#!/usr/bin/env python3

from marpdan import *
from marpdan.problems import *
from marpdan.baselines import *
from marpdan.externals import *
from marpdan.dep import *
from marpdan.utils import *
from marpdan.layers import reinforce_loss

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import time
import os
from itertools import chain
import numpy as np
import random


def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    with tqdm(loader, desc="Ep.#{: >3d}/{: <3d}".format(ep + 1, args.epoch_count)) as progress:
        for minibatch in progress:
            if data.cust_mask is None:
                custs, mask = minibatch.to(device), None
            else:
                custs, mask = minibatch[0].to(device), minibatch[1].to(device)

            dyna = Environment(data, custs, mask, *env_params)
            actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
            loss = reinforce_loss(logps, rewards, bl_vals)

            prob = torch.stack(logps).sum(0).exp().mean()
            # val = torch.stack(rewards).sum(0).mean()
            val = rewards.mean()
            bl = bl_vals[0].mean()

            optim.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                                            args.max_grad_norm)
            optim.step()

            progress.set_postfix_str("l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                loss, prob, val, bl, grad_norm))

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item()
            ep_norm += grad_norm

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))


def test_epoch(args, test_env, learner, ref_costs):
    learner.eval()
    if args.problem_type[0] == "s":
        costs = test_env.nodes.new_zeros(test_env.minibatch_size)
        for _ in range(100):
            _, _, rewards = learner(test_env)
            costs -= torch.stack(rewards).sum(0).squeeze(-1)
        costs = costs / 100
    else:
        _, _, rs = learner(test_env)
        costs = -torch.stack(rs).sum(dim=0).squeeze(-1)
    mean = costs.mean()
    std = costs.std()
    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()


def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.rng_seed is not None:
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
        torch.cuda.manual_seed(args.rng_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.rng_seed)

    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs):
            pass

    # PROBLEM
    Dataset = {
        "vrp": VRP_Dataset,
        "vrptw": VRPTW_Dataset,
        "svrptw": VRPTW_Dataset,
        "sdvrptw": SDVRPTW_Dataset
    }.get(args.problem_type)
    gen_params = [
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.dem_range
    ]
    if args.problem_type != "vrp":
        gen_params.extend([args.horizon, args.dur_range, args.tw_ratio, args.tw_range])
    if args.problem_type == "sdvrptw":
        gen_params.extend([args.deg_of_dyna, args.appear_early_ratio])

    # TRAIN DATA
    verbose_print("Generating {} {} samples of training data...".format(
        args.iter_count * args.batch_size, args.problem_type.upper()),
        end=" ", flush=True)
    train_data = Dataset.generate(
        args.iter_count * args.batch_size,
        *gen_params, new_dy=True
    )
    train_data.normalize()
    verbose_print("Done.")

    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end=" ", flush=True)
    test_data = Dataset.generate(
        args.test_batch_size,
        *gen_params, new_dy=True
    )
    verbose_print("Done.")

    #  用求解器求解测试集
    if ORTOOLS_ENABLED:
        ref_routes = ort_solve(test_data)
    elif LKH_ENABLED:
        ref_routes = lkh_solve(test_data)
    else:
        ref_routes = None
        print("Warning! No external solver found to compute gaps for test.")
    test_data.normalize()

    # ENVIRONMENT
    Environment = {
        "vrp": VRP_Environment,
        "vrptw": VRPTW_Environment,
        "svrptw": SVRPTW_Environment,
        "sdvrptw": SDVRPTW_Environment
    }.get(args.problem_type)
    env_params = [args.pending_cost]
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type != "vrptw":
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])
    test_env = Environment(test_data, None, None, *env_params)

    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == 's' else 1)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    test_env.nodes = test_env.nodes.to(dev)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    # MODEL
    verbose_print("Initializing attention model...",
                  end=" ", flush=True)
    # 完整模型
    learner = AttentionLearner(
        Dataset.CUST_FEAT_SIZE,
        Environment.VEH_STATE_SIZE,
        args.model_size,
        args.layer_count,
        args.head_count,
        args.ff_size,
        args.tanh_xplor
    )
    learner.to(dev)
    verbose_print("Done.")

    # BASELINE
    verbose_print("Initializing '{}' baseline...".format(
        args.baseline_type),
        end=" ", flush=True)
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
    baseline.to(dev)
    verbose_print("Done.")

    # OPTIMIZER AND LR SCHEDULER
    verbose_print("Initializing Adam optimizer...",
                  end=" ", flush=True)
    lr_sched = None
    if args.baseline_type == "critic":
        optim = Adam([
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate}
        ])
        if args.rate_decay is not None:
            _decay = args.rate_decay if args.critic_decay is None else args.critic_decay
            lr_sched = LambdaLR(optim, [
                lambda ep: args.learning_rate * args.rate_decay ** ep,
                lambda ep: args.critic_rate * critic_decay ** ep
            ])
    else:
        optim = Adam(learner.parameters(), args.learning_rate)
        if args.rate_decay is not None:
            lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay ** ep)
    verbose_print("Done.")

    # CHECKPOINTING
    verbose_print("Creating output dir...",
                  end=" ", flush=True)
    args.output_dir = "./output/{}n{}m{}_{}".format(
        args.problem_type.upper(),
        args.customers_count,
        args.vehicles_count,
        time.strftime("%y%m%d-%H%M")
    ) if args.output_dir is None else args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))

    if args.resume_state is None:
        start_ep = 0
    else:
        start_ep = load_checkpoint(args, learner, optim, baseline, lr_sched)

    verbose_print("Running...")
    train_stats = []
    test_stats = []
    try:
        for ep in range(start_ep, args.epoch_count):
            train_stats.append(train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep))
            if ref_routes is not None:
                test_stats.append(test_epoch(args, test_env, learner, ref_costs))

            if args.rate_decay is not None:
                lr_sched.step()
            if args.pend_cost_growth is not None:
                env_params[0] *= args.pend_cost_growth
            if args.late_cost_growth is not None:
                env_params[1] *= args.late_cost_growth
            if args.grad_norm_decay is not None:
                args.max_grad_norm *= args.grad_norm_decay

            if (ep + 1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

    except KeyboardInterrupt:
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)


if __name__ == "__main__":
    main(parse_args())
