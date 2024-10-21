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
import copy
from collections import OrderedDict
import random
import numpy as np
import math


def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    for minibatch in loader:
        dyna = get_env(Environment, data, minibatch, device, env_params)
        actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
        loss = reinforce_loss(logps, rewards, bl_vals)

        prob = torch.stack(logps).sum(0).exp().mean()
        if type(rewards) == list:
            val = torch.stack(rewards).sum(0).mean()
        else:
            val = rewards.mean()
        bl = bl_vals[0].mean()

        optim.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                                        args.max_grad_norm)
        optim.step()

        # progress.set_postfix_str("l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
        #     loss, prob, val, bl, grad_norm))

        ep_loss += loss.item()
        ep_prob += prob.item()
        ep_val += val.item()
        ep_bl += bl.item()
        ep_norm += grad_norm

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))


def train_epoch_maml(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep, fast_weight):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)
    # print(f"max--{torch.cuda.max_memory_allocated()}")  # 显示显存
    # print(f"cur--{torch.cuda.memory_allocated()}")
    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    for minibatch in loader:
        dyna = get_env(Environment, data, minibatch, device, env_params)  # 加显存
        fast_weight = load_model(bl_wrapped_learner, fast_weight)
        actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)  # 加极多显存
        loss = reinforce_loss(logps, rewards, bl_vals)
        prob = torch.stack(logps).sum(0).exp().mean()
        if type(rewards) == list:
            val = torch.stack(rewards).sum(0).mean()
        else:
            val = rewards.mean()
        bl = bl_vals[0].mean()
        gradients = torch.autograd.grad(loss, fast_weight.values(), create_graph=True)
        w_t, (beta1, beta2), eps = [], optim.param_groups[0]['betas'], optim.param_groups[0]['eps']
        lr, weight_decay = args.inner_learning_rate, args.inner_rate_decay if args.inner_rate_decay else 1
        for i, ((name, param), grad) in enumerate(zip(fast_weight.items(), gradients)):
            if optim.state_dict()['state'] != {}:
                state = optim.state_dict()['state'][i]
                step, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']
                step += 1
                step = step.item() if isinstance(step, torch.Tensor) else step
                grad = grad.add(param, alpha=weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                param = param - step_size * exp_avg / denom
                optim.state_dict()['state'][i]['exp_avg'] = exp_avg.clone().detach()
                optim.state_dict()['state'][i]['exp_avg_sq'] = exp_avg_sq.clone().detach()
            else:
                param = param - lr * grad
            w_t.append((name, param))
        fast_weight = OrderedDict(w_t)
        if args.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                                        args.max_grad_norm)
        # optim.step()

        # progress.set_postfix_str("l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
        #     loss, prob, val, bl, grad_norm))

        ep_loss += loss.item()
        ep_prob += prob.item()
        ep_val += val.item()
        ep_bl += bl.item()
        ep_norm += grad_norm

    return tuple(stat / args.k for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm)), fast_weight


def fast_val(args, data, Environment, env_params, bl_wrapped_learner, fast_weight):
    if fast_weight:
        fast_weight = load_model(bl_wrapped_learner, fast_weight)
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    # 由于val_data的大小与batch_size相同，该循环只进行一次
    for minibatch in loader:
        dyna = get_env(Environment, data, minibatch, 'cuda', env_params)
        actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
        loss = reinforce_loss(logps, rewards, bl_vals)

    return loss


def load_model(baseline, weight):
    for name, param in weight.items():
        if name in baseline.learner.state_dict():
            baseline.learner.state_dict()[name].copy_(param)
        else:
            baseline.project.state_dict()[name].copy_(param)

    return OrderedDict(**OrderedDict(baseline.learner.named_parameters()),
                       **OrderedDict(baseline.project.named_parameters()))


def test_epoch(args, test_env, learner, ref_costs):
    with torch.no_grad():
        learner.eval()
        costs = test_env.nodes.new_zeros(test_env.minibatch_size)
        for _ in range(100):
            _, _, rewards = learner(test_env)
            costs -= torch.stack(rewards).sum(0).squeeze(-1)
        costs = costs / 100
        mean = costs.mean()
        std = costs.std()
        gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()


def get_data(data_num, data_type, *gen_params):
    Dataset = {
        "tj": DVRPTW_TJ_Dataset,
        "qs": DVRPTW_QS_Dataset,
        "qc": DVRPTW_QC_Dataset,
        "vb": DVRPTW_VB_Dataset,
        "nr": DVRPTW_NR_Dataset
    }.get(data_type)
    train_data = Dataset.generate(data_num, *gen_params)
    train_data.normalize()
    return train_data


def get_env(Environment, data, minibatch, device, env_params):
    if Environment == DVRPTW_TJ_Environment:
        if data.cust_mask is None:
            custs, mask, tj_time, distances_with_tj = minibatch[0].to(device), None, minibatch[1].to(device), \
                minibatch[2].to(device)
        else:
            custs, mask, tj_time, distances_with_tj = minibatch[0].to(device), minibatch[1].to(device), \
                minibatch[
                    2].to(device), minibatch[3].to(device)
        dyna = Environment(data, custs, mask, tj_time, distances_with_tj, *env_params)
    elif Environment == DVRPTW_QS_Environment:
        if data.cust_mask is None:
            custs, mask, changed_dem = minibatch[0].to(device), None, minibatch[1].to(device)
        else:
            custs, mask, changed_dem = minibatch[0].to(device), minibatch[1].to(device), minibatch[2].to(
                device)
        dyna = Environment(data, custs, mask, changed_dem, *env_params)
    elif Environment == DVRPTW_QC_Environment or Environment == DVRPTW_NR_Environment:
        if data.cust_mask is None:
            custs, mask = minibatch.to(device), None
        else:
            custs, mask = minibatch[0].to(device), minibatch[1].to(device)
        dyna = Environment(data, custs, mask, *env_params)
    elif Environment == DVRPTW_VB_Environment:
        if data.cust_mask is None:
            custs, b_veh_idx, b_time, mask = minibatch[0].to(device), minibatch[1].to(device), minibatch[2].to(
                device), None
        else:
            custs, b_veh_idx, b_time, mask = minibatch[0].to(device), minibatch[1].to(device), minibatch[2].to(
                device), minibatch[3].to(device)
        dyna = Environment(data, custs, mask, b_veh_idx, b_time, *env_params)
    else:
        raise NotImplementedError
    return dyna


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
        "tj": DVRPTW_TJ_Dataset,
        "qs": DVRPTW_QS_Dataset,
        "qc": DVRPTW_QC_Dataset,
        "vb": DVRPTW_VB_Dataset,
        "nr": DVRPTW_NR_Dataset
    }.get(args.problem_type)
    test_gen_params = [
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.dem_range,
        args.horizon,
        args.dur_range,
        args.tw_r,
        args.tw_range,
        args.deg_of_dyna,
        args.appear_early_ratio,
    ]

    # TRAIN DATA
    # verbose_print("Generating {} {} samples of training data...".format(
    #     args.iter_count * args.batch_size, args.problem_type.upper()),
    #     end=" ", flush=True)
    # train_data = Dataset.generate(
    #     args.iter_count * args.batch_size,
    #     *gen_params
    # )
    # train_data.normalize()
    # verbose_print("Done.")

    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end=" ", flush=True)
    test_data = Dataset.generate(
        args.test_batch_size,
        *test_gen_params
    )
    verbose_print("Done.")

    #  用求解器求解测试集
    if args.ref_enable:
        if ORTOOLS_ENABLED:
            ref_routes = ort_solve(test_data)
        elif LKH_ENABLED:
            ref_routes = lkh_solve(test_data)
        else:
            ref_routes = None
            print("Warning! No external solver found to compute gaps for test.")
    else:
        ref_routes = None
    test_data.normalize()

    # ENVIRONMENT
    Environment = {
        "tj": DVRPTW_TJ_Environment,
        "qs": DVRPTW_QS_Environment,
        "qc": DVRPTW_QC_Environment,
        "vb": DVRPTW_VB_Environment,
        "nr": DVRPTW_NR_Environment
    }.get(args.problem_type)
    env_params = [args.pending_cost, args.late_cost, args.speed_var, args.late_prob, args.slow_down, args.late_var]
    if args.problem_type == "tj":
        test_env = Environment(test_data, None, None, None, None, *env_params)
        test_env.tj_time = test_env.tj_time.to(dev)
        test_env.distances_with_tj = test_env.distances_with_tj.to(dev)
    elif args.problem_type == "qs":
        test_env = Environment(test_data, None, None, None, *env_params)
        test_env.changed_dem = test_env.changed_dem.to(dev)
    elif args.problem_type == "qc" or args.problem_type == "nr":
        test_env = Environment(test_data, None, None, *env_params)
    elif args.problem_type == 'vb':
        test_env = Environment(test_data, None, None, None, None, *env_params)
        test_env.b_veh_idx = test_data.b_veh_idx.to(dev)
        test_env.b_time = test_data.b_time.to(dev)
    else:
        print("Unrecognized problem type!")
        raise NotImplementedError
    # 放入cuda比较重要
    test_env.nodes = test_env.nodes.to(dev)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == 's' else 1)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    print(f"max--{torch.cuda.max_memory_allocated()}")  # 显示显存
    print(f"cur--{torch.cuda.memory_allocated()}")
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
        if args.rate_decay is not None:
            optim = Adam([
                {"params": baseline.learner.parameters(), "lr": args.learning_rate, "weight_decay": args.rate_decay},
                {"params": baseline.parameters(), "lr": args.critic_rate, "weight_decay": args.critic_decay}
            ])
        else:
            optim = Adam([
                {"params": baseline.learner.parameters(), "lr": args.learning_rate},
                {"params": baseline.parameters(), "lr": args.critic_rate}
            ])
        # if args.rate_decay is not None:
        #     _decay = args.rate_decay if args.critic_decay is None else args.critic_decay
        #     lr_sched = LambdaLR(optim, [
        #         lambda ep: args.learning_rate * args.rate_decay ** ep,
        #         lambda ep: args.critic_rate * args.critic_decay ** ep
        #     ])
    else:
        if args.rate_decay is not None:
            optim = Adam(baseline.learner.parameters(), args.learning_rate, weight_decay=args.rate_decay)
        else:
            optim = Adam(baseline.learner.parameters(), args.learning_rate)
        # if args.rate_decay is not None:
        #     lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay ** ep)
    verbose_print("Done.")

    # CHECKPOINTING
    verbose_print("Creating output dir...",
                  end=" ", flush=True)
    args.output_dir = "./output/{}n{}m{}_{}".format(
        'meta' if args.meta_enable else 'nometa',
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
    # TRAINING
    verbose_print("Running...")
    bt = time.time()
    train_stats = []
    test_stats = []
    max_dod = args.max_dod
    min_dod = args.min_dod
    dod_interval = args.dod_interval
    problem_type_num = args.problem_type_num
    group_num = int((max_dod - min_dod) // dod_interval + 1)
    group_size = (args.epoch_count - start_ep + 1) // group_num
    task_w = torch.full((group_num, problem_type_num), 1 / problem_type_num)
    # task_types = ['nr', 'vb', 'qc', 'qs', 'tj']
    task_types = ['nr', 'qc', 'tj']
    try:
        if args.meta_enable:
            for ep in tqdm(range(start_ep, args.epoch_count), desc="Epochs"):
                optim.zero_grad()
                fast_weights, val_loss, meta_grad_dict = [], 0, {(i, j): 0 for i, group in
                                                                 enumerate(optim.param_groups) for j, _ in
                                                                 enumerate(group['params'])}
                args.alpha = alpha_scheduler(args.alpha, args.alpha_decay)
                fast_learner_weights, fast_project_weights, selected_tasks = [], [], []
                idx = int((ep - start_ep) // group_size)
                weight = task_w[idx]
                if args.curriculum:
                    cur_dod = min_dod + idx * dod_interval
                else:
                    cur_dod = min_dod + random.randint(0, group_num - 1) * dod_interval
                gen_params = [
                    args.customers_count,
                    args.vehicles_count,
                    args.veh_capa,
                    args.veh_speed,
                    args.min_cust_count,
                    args.loc_range,
                    args.dem_range,
                    args.horizon,
                    args.dur_range,
                    args.tw_ratio,
                    args.tw_range,
                    cur_dod,
                    args.appear_early_ratio,
                ]
                if ep != 0 and ep % args.update_interval == 0:
                    task_w[idx] = update_task_weight(learner=baseline.learner, data_num=args.batch_size, weights=weight,
                                                     env_params=env_params, gen_params=gen_params,
                                                     task_types=task_types)
                for pb in range(args.B):
                    selected = torch.multinomial(weight, 1).item()
                    selected_tasks.append(task_types[selected])
                fast_weight = None
                for b in range(args.B):
                    if args.curriculum:
                        task_type = selected_tasks[b]
                    else:
                        task_type = task_types[random.randint(0, len(task_types) - 1)]
                    if args.meta_method in ['fomaml', 'reptile']:
                        # 可能存在问题，将fn_grad不为None的tensor置None
                        baseline.learner.fleet_attention._k_proj = None
                        baseline.learner.fleet_attention._v_proj = None
                        baseline.learner.combine_attention._k_proj = None
                        baseline.learner.combine_attention._v_proj = None
                        baseline.learner.cust_repr = None
                        task_baseline = copy.deepcopy(baseline)
                        task_optim = Adam([
                            {"params": task_baseline.learner.parameters(), "lr": args.learning_rate},
                            {"params": task_baseline.parameters(), "lr": args.critic_rate}
                        ])
                    elif args.meta_method == 'maml':
                        if not fast_weight:
                            fast_weight = OrderedDict(**OrderedDict(baseline.learner.named_parameters()),
                                                      **OrderedDict(baseline.project.named_parameters()))
                            ori_fast_weight = copy.deepcopy(fast_weight)
                        else:
                            fast_weight = copy.deepcopy(ori_fast_weight)
                    Environment = {
                        "tj": DVRPTW_TJ_Environment,
                        "qs": DVRPTW_QS_Environment,
                        "qc": DVRPTW_QC_Environment,
                        "vb": DVRPTW_VB_Environment,
                        "nr": DVRPTW_NR_Environment
                    }.get(task_type)
                    # inner loop, 在train_epoch中包含循环, 这里k的处理方式改变了
                    data = get_data(args.batch_size, task_type, *gen_params)
                    for i in range(args.k):
                        if args.meta_method in ['fomaml', 'reptile']:
                            train_state = train_epoch(args, data, Environment, env_params, task_baseline, task_optim,
                                                      dev, ep)
                        elif args.meta_method == 'maml':
                            train_state, fast_weight = train_epoch_maml(args, data, Environment, env_params,
                                                                        baseline, optim, dev, ep, fast_weight)
                    # val
                    if args.meta_method == 'maml':
                        val_data = get_data(args.batch_size, task_type, *gen_params)
                        # 这里数据的多少可能需要调整
                        loss = fast_val(args, val_data, Environment, env_params, baseline, fast_weight)
                        optim.zero_grad()
                        loss.backward()
                        for i, group in enumerate(optim.param_groups):
                            for j, p in enumerate(group['params']):
                                meta_grad_dict[(i, j)] += p.grad
                        # _ = load_model(baseline, ori_fast_weight)
                    elif args.meta_method == 'fomaml':
                        val_data = get_data(args.batch_size, task_type, *gen_params)
                        loss = fast_val(args, val_data, Environment, env_params, task_baseline, None)
                        task_optim.zero_grad()
                        loss.backward()
                        for i, group in enumerate(task_optim.param_groups):
                            for j, p in enumerate(group['params']):
                                meta_grad_dict[(i, j)] += p.grad
                    elif args.meta_method == 'reptile':
                        fast_learner_weights.append(task_baseline.learner.state_dict())
                        fast_project_weights.append(task_baseline.project.state_dict())
                    else:
                        raise NotImplementedError
                # outer loop
                if args.meta_method == 'maml':
                    _ = load_model(baseline, ori_fast_weight)
                    optim.zero_grad()
                    for i, group in enumerate(optim.param_groups):
                        for j, p in enumerate(group['params']):
                            p.grad = meta_grad_dict[(i, j)]
                    optim.step()
                elif args.meta_method == 'fomaml':
                    optim.zero_grad()
                    for i, group in enumerate(optim.param_groups):
                        for j, p in enumerate(group['params']):
                            p.grad = meta_grad_dict[(i, j)]
                    optim.step()
                elif args.meta_method == 'reptile':
                    state_dict = {
                        params_key: (learner.state_dict()[params_key] + args.alpha * torch.mean(torch.stack(
                            [fast_weight[params_key] - learner.state_dict()[params_key] for fast_weight in
                             fast_learner_weights],
                            dim=0).float(), dim=0)) for params_key in learner.state_dict()}
                    learner.load_state_dict(state_dict)

                    state_dict = {
                        params_key: (baseline.project.state_dict()[params_key] + args.alpha * torch.mean(torch.stack(
                            [fast_weight[params_key] - baseline.project.state_dict()[params_key] for fast_weight in
                             fast_project_weights],
                            dim=0).float(), dim=0)) for params_key in baseline.project.state_dict()}
                    baseline.project.load_state_dict(state_dict)

                train_stats.append(train_state)
                if ref_routes is not None and not ep % args.val_interval:
                    test_stats.append(test_epoch(args, test_env, learner, ref_costs))

                # if args.rate_decay is not None:
                #     lr_sched.step()
                if args.pend_cost_growth is not None:
                    env_params[0] *= args.pend_cost_growth
                if args.late_cost_growth is not None:
                    env_params[1] *= args.late_cost_growth
                if args.grad_norm_decay is not None:
                    args.max_grad_norm *= args.grad_norm_decay

                if (ep + 1) % args.checkpoint_period == 0:
                    save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
        else:
            # 不使用元学习
            for ep in tqdm(range(start_ep, args.epoch_count), desc="Epochs"):
                idx = int((ep - start_ep) // group_size)
                if args.curriculum:
                    cur_dod = min_dod + idx * dod_interval
                else:
                    cur_dod = min_dod + random.randint(0, group_num - 1) * dod_interval
                gen_params = [
                    args.customers_count,
                    args.vehicles_count,
                    args.veh_capa,
                    args.veh_speed,
                    args.min_cust_count,
                    args.loc_range,
                    args.dem_range,
                    args.horizon,
                    args.dur_range,
                    args.tw_ratio,
                    args.tw_range,
                    cur_dod,
                    args.appear_early_ratio,
                ]
                task_type = task_types[random.randint(0, len(task_types) - 1)]
                data = get_data(args.batch_size, task_type, *gen_params)
                Environment = {
                    "tj": DVRPTW_TJ_Environment,
                    "qs": DVRPTW_QS_Environment,
                    "qc": DVRPTW_QC_Environment,
                    "vb": DVRPTW_VB_Environment,
                    "nr": DVRPTW_NR_Environment
                }.get(task_type)
                # print(f"max--{torch.cuda.max_memory_allocated()}")  # 显示显存
                # print(f"cur--{torch.cuda.memory_allocated()}")
                train_stats.append(train_epoch(args, data, Environment, env_params, baseline, optim, dev, ep))
                if ref_routes is not None:
                    test_stats.append(test_epoch(args, test_env, learner, ref_costs))

                # if args.rate_decay is not None:
                #     lr_sched.step()
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
        # 如果报错是因为test_env中一些数据不在cuda
        export_train_test_stats(args, start_ep, train_stats, test_stats)
        print(f'total training time: {time.time() - bt}')


def update_task_weight(learner, data_num, weights, env_params, gen_params, task_types):
    with torch.no_grad():
        # global run_func
        start_t, gap = time.time(), torch.zeros(weights.size(0))
        # batch_size = 200 if self.meta_params["solver"] == "lkh3_offline" else 50
        # idx = torch.randperm(batch_size)[:50]
        gaps = []
        qos = []
        for i in task_types:
            Environment = {
                "tj": DVRPTW_TJ_Environment,
                "qs": DVRPTW_QS_Environment,
                "qc": DVRPTW_QC_Environment,
                "vb": DVRPTW_VB_Environment,
                "nr": DVRPTW_NR_Environment
            }.get(i)
            data = get_data(data_num, i, *gen_params)
            if i == "tj":
                test_env = Environment(data, None, None, None, None, *env_params)
                test_env.tj_time = test_env.tj_time.to('cuda')
                test_env.distances_with_tj = test_env.distances_with_tj.to('cuda')
            elif i == "qs":
                test_env = Environment(data, None, None, None, *env_params)
                test_env.changed_dem = test_env.changed_dem.to('cuda')
            elif i == "qc" or i == "nr":
                test_env = Environment(data, None, None, *env_params)
            elif i == 'vb':
                test_env = Environment(data, None, None, None, None, *env_params)
                test_env.b_veh_idx = data.b_veh_idx.to('cuda')
                test_env.b_time = data.b_time.to('cuda')
            else:
                print("Unrecognized problem type!")
                raise NotImplementedError
            test_env.nodes = test_env.nodes.to('cuda')
            if test_env.init_cust_mask is not None:
                test_env.init_cust_mask = test_env.init_cust_mask.to('cuda')
            # tools_costs = eval_apriori_routes(test_env, ort_solve(data), 100)
            _, _, rewards = learner(test_env)
            pending = (test_env.served ^ True).float().sum(-1) - 1
            gaps.append((pending / (test_env.nodes_count - 1)).mean())   # 改了
        print(">> Finish updating task weights within {}s".format(round(time.time() - start_t, 2)))
        temp = 1.0
        gap_temp = torch.Tensor([i / temp for i in gaps])
        print(gaps, temp)
        print(">> Old task weights: {}".format(weights))
        weights = torch.softmax(gap_temp, dim=0)
        print(">> New task weights: {}".format(weights))

    return weights


def alpha_scheduler(alpha, alpha_decay):
    """
    Update param for Reptile.
    """
    return max(alpha * alpha_decay, 0.0001)


if __name__ == "__main__":
    main(parse_args())
