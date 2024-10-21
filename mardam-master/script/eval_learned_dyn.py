import time

from marpdan import AttentionLearner
from marpdan.problems import *
from marpdan.dep import tqdm

import torch
from torch.utils.data import DataLoader
from marpdan.layers import reinforce_loss
import os
from torch.nn.utils import clip_grad_norm_
from itertools import chain
from marpdan.baselines import *
from torch.optim import Adam
from marpdan.utils import *
import random
import numpy as np
import contextlib
import io


def get_data(data_num, Dataset, *gen_params):
    train_data = Dataset.generate(data_num, *gen_params)
    train_data.normalize()
    return train_data


def save_checkpoint(output_dir, fine_tune_step, learner, optim, baseline=None):
    checkpoint = {
        "fine_tune_step": fine_tune_step,
        "model": learner.state_dict(),
        "optim": optim.state_dict()
    }
    if type(baseline) == CriticBaseline:
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, output_dir)


def load_checkpoint(args, learner, optim, baseline=None, lr_sched=None):
    checkpoint = torch.load(args.resume_state)
    learner.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic":
        baseline.load_state_dict(checkpoint["critic"])
    return checkpoint["ep"]


def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)
    costs = []
    qos = []
    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    for minibatch in loader:
        if Environment == DVRPTW_TJ_Environment or Environment == DVRPTW_NR_TJ_Environment:
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
        elif Environment == DVRPTW_QS_VB_Environment:
            if data.cust_mask is None:
                custs, mask, cd, b_veh_idx, b_time = minibatch[0].to(device), None, minibatch[1].to(device), minibatch[2].to(
                    device), minibatch[3].to(device)
            else:
                custs, mask, cd, b_veh_idx, b_time  = minibatch[0].to(device), minibatch[1].to(device), minibatch[2].to(
                    device), minibatch[3].to(device), minibatch[4].to(device)
            dyna = Environment(data, custs, mask, cd, b_veh_idx, b_time, *env_params)
        else:
            raise NotImplementedError
        actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
        loss = reinforce_loss(logps, rewards, bl_vals)

        prob = torch.stack(logps).sum(0).exp().mean()
        if type(rewards) == list:
            if not isinstance(rewards[0], torch.Tensor):
                rewards = [t[0] for t in rewards]
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
        costs.append(-torch.stack(rewards).sum(0).squeeze(-1))
        pending = (dyna.served ^ True).float().sum(-1) - 1
        qos.append(1 - pending / (dyna.nodes_count - 1))

    costs = torch.cat(costs, 0)
    qos = torch.cat(qos, 0)

    return costs, qos


SEED = 1234
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(SEED)
args = parse_args()
problem_size = 100
## deg_of_dyna = 0.3
deg_of_dynas = [0.3, 0.4, 0.6]
fine_tune_enable = True
appear_early_ratio = 0.5
baseline_type = 'critic'
fine_tune_steps = 10000
is_vrplib = False
chkpt_set = ["./output/fNo_{}_main3/chkpt_ep40000.pyth".format(problem_size),
             "./output/fomaml_{}_main3/chkpt_ep20000.pyth".format(problem_size)]
# ['nr', 'qc', 'tj', 'qs', 'vb', 'nr_tj', 'qs_vb']
print(f"fine tune steps: {fine_tune_steps}")
for chkpt_path in chkpt_set:
    print("=======================================Load chkpt from {}".format(chkpt_path))
    for problem_type in ['nr', 'qc', 'tj', 'qs_vb']:
        for deg_of_dyna in deg_of_dynas:
            print("====begin {} dod{} data test:".format(problem_type, deg_of_dyna))
            # chkpt_path = "./output/metan50m10_0/chkpt_ep4000.pyth"
            # chkpt_path = "./output/nometan50m10_0/chkpt_ep20.pyth"
            # chkpt_path = "./results/tmp/best_ckp.pyth"
            # chkpt_path = "./results/vb_n20m4_0.4_0.5_1/best_ckp.pyth"
            Dataset = {
                "tj": DVRPTW_TJ_Dataset,
                "qs": DVRPTW_QS_Dataset,
                "qc": DVRPTW_QC_Dataset,
                "vb": DVRPTW_VB_Dataset,
                "nr": DVRPTW_NR_Dataset,
                "nr_tj": DVRPTW_NR_TJ_Dataset,
                "qs_vb": DVRPTW_QS_VB_Dataset
            }.get(problem_type)
            Environment = {
                "tj": DVRPTW_TJ_Environment,
                "qs": DVRPTW_QS_Environment,
                "qc": DVRPTW_QC_Environment,
                "vb": DVRPTW_VB_Environment,
                "nr": DVRPTW_NR_Environment,
                "nr_tj": DVRPTW_NR_TJ_Environment,
                "qs_vb": DVRPTW_QS_VB_Environment
            }.get(problem_type)
            # 模型
            chkpt = torch.load(chkpt_path, map_location='cpu')
            learner = AttentionLearner(7, 4)
            learner.greedy = True
            learner = learner.to('cuda')
            learner.load_state_dict(chkpt["model"])
            # 测试数据
            if not is_vrplib:
                data = torch.load(
                    "./data/{}_n{}m{}_{}_{}_10000/norm_data.pyth".format(problem_type, problem_size, problem_size // 5,
                                                                         deg_of_dyna,
                                                                         appear_early_ratio))
            else:
                data = Dataset.read_instance("./data/B-VRP/B-n63-k10.vrp")
                data.normalize()
            assert type(data) == Dataset
            data.nodes = data.nodes.to('cuda')
            if Dataset == DVRPTW_NR_Dataset or Dataset == DVRPTW_QC_Dataset:
                pass
            elif Dataset == DVRPTW_VB_Dataset:
                data.b_veh_idx = data.b_veh_idx.to('cuda')
                data.b_time = data.b_time.to('cuda')
            elif Dataset == DVRPTW_QS_Dataset:
                data.changed_dem = data.changed_dem.to('cuda')
            elif Dataset == DVRPTW_TJ_Dataset or Dataset == Dataset == DVRPTW_NR_TJ_Dataset:
                data.tj_time = data.tj_time.to('cuda')
                data.distances_with_tj = data.distances_with_tj.to('cuda')
            elif Dataset == DVRPTW_QS_VB_Dataset:  # 改了
                data.changed_dem = data.changed_dem.to('cuda')
                data.b_veh_idx = data.b_veh_idx.to('cuda')
                data.b_time = data.b_time.to('cuda')
            else:
                raise NotImplementedError
            loader = DataLoader(data, batch_size=50)
            bt = time.time()
            if not fine_tune_enable:
                costs = []
                qos = []
                learner.eval()
                with torch.no_grad():
                    for b, batch in enumerate(tqdm(loader)):
                        if Dataset == DVRPTW_NR_Dataset or Dataset == DVRPTW_QC_Dataset:
                            env = Environment(data, batch, pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_VB_Dataset or Dataset == DVRPTW_TJ_Dataset or Dataset == DVRPTW_NR_TJ_Dataset:
                            env = Environment(data, batch[0], None, batch[1], batch[2], pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_QS_Dataset:
                            env = Environment(data, batch[0], None, batch[1], pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_QS_VB_Dataset:
                            env = Environment(data, batch[0], None, batch[1], batch[2], batch[3], pending_cost=0, late_p=0)
                        else:
                            raise NotImplementedError
                        _, _, rewards = learner(env)
                        costs.append(-torch.stack(rewards).sum(0).squeeze(-1))
                        pending = (env.served ^ True).float().sum(-1) - 1
                        qos.append(1 - pending / (env.nodes_count - 1))

                    costs = torch.cat(costs, 0)
                    qos = torch.cat(qos, 0)

                    print("{:5.2f} +- {:5.2f} (qos={:.2%})".format(costs.mean(), costs.std(),
                                                                   qos.mean()))
            else:
                args.baseline_type = baseline_type
                args.resume_state = chkpt_path
                # 应与测试集相同
                gen_params = [
                    problem_size,
                    problem_size//5,
                    args.veh_capa,
                    args.veh_speed,
                    args.min_cust_count,
                    args.loc_range,
                    args.dem_range,
                    args.horizon,
                    args.dur_range,
                    args.tw_ratio,
                    args.tw_range,
                    deg_of_dyna,
                    appear_early_ratio,
                ]
                # baseline
                if baseline_type == "none":
                    baseline = NoBaseline(learner)
                elif baseline_type == "nearnb":
                    baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
                elif baseline_type == "rollout":
                    loss_use_cumul = True
                    baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
                elif baseline_type == "critic":
                    baseline = CriticBaseline(learner, problem_size, args.critic_use_qval, args.loss_use_cumul)
                    baseline.load_state_dict(chkpt["critic"])
                else:
                    raise NotImplementedError
                baseline.to('cuda')
                # 优化器
                if args.baseline_type == "critic":
                    if args.rate_decay is not None:
                        task_optim = Adam([
                            {"params": baseline.learner.parameters(), "lr": args.learning_rate,
                             "weight_decay": args.rate_decay},
                            {"params": baseline.parameters(), "lr": args.critic_rate, "weight_decay": args.critic_decay}
                        ])
                    else:
                        task_optim = Adam([
                            {"params": baseline.learner.parameters(), "lr": args.learning_rate},
                            {"params": baseline.parameters(), "lr": args.critic_rate}
                        ])
                else:
                    if args.rate_decay is not None:
                        task_optim = Adam(baseline.learner.parameters(), args.learning_rate, weight_decay=args.rate_decay)
                    else:
                        task_optim = Adam(baseline.learner.parameters(), args.learning_rate)
                load_checkpoint(args, learner, task_optim, baseline)
                env_params = [args.pending_cost, args.late_cost, args.speed_var, args.late_prob, args.slow_down,
                              args.late_var]
                out_dir = "./results/{}_n{}m{}_{}_{}_{}/".format(problem_type, problem_size, problem_size // 5, deg_of_dyna,
                                                                 appear_early_ratio, time.strftime("%y%m%d-%H%M"))
                os.makedirs(os.path.dirname(out_dir), exist_ok=True)
                args.epoch_count = fine_tune_steps
                best_cost = 1000.0
                f = io.StringIO()
                for ep in range(fine_tune_steps):
                    if problem_size > 50:
                        args.batch_size = 256
                    data = get_data(args.batch_size, Dataset, *gen_params)
                    data.nodes = data.nodes.to('cuda')
                    costs, qos = train_epoch(args, data, Environment, env_params, baseline, task_optim, 'cuda', ep)
                    if costs.mean() < best_cost:
                        best_cost = costs.mean()
                        save_checkpoint(os.path.join(out_dir, "best_ckp.pyth"), ep, learner, task_optim, baseline)

                    torch.save({"costs": costs, "qos": qos}, os.path.join(out_dir, "mardan.pyth"))

                save_checkpoint(os.path.join(out_dir, "chkpt_fine_tune_step{}.pyth".format(fine_tune_steps)),
                                fine_tune_steps,
                                learner, task_optim, baseline)

                costs = []
                qos = []
                learner.eval()
                with torch.no_grad():
                    for b, batch in enumerate(tqdm(loader)):
                        if Dataset == DVRPTW_NR_Dataset or Dataset == DVRPTW_QC_Dataset:
                            env = Environment(data, batch, pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_VB_Dataset or Dataset == DVRPTW_TJ_Dataset or Dataset == DVRPTW_NR_TJ_Dataset:
                            env = Environment(data, batch[0], None, batch[1], batch[2], pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_QS_Dataset:
                            env = Environment(data, batch[0], None, batch[1], pending_cost=0, late_p=0)
                        elif Dataset == DVRPTW_QS_VB_Dataset:
                            env = Environment(data, batch[0], None, batch[1], batch[2], batch[3], pending_cost=0, late_p=0)
                        else:
                            raise NotImplementedError
                        _, _, rewards = learner(env)
                        costs.append(-torch.stack(rewards).sum(0).squeeze(-1))
                        pending = (env.served ^ True).float().sum(-1) - 1
                        qos.append(1 - pending / (env.nodes_count - 1))

                    costs = torch.cat(costs, 0)
                    qos = torch.cat(qos, 0)

                    print("{:5.2f} +- {:5.2f} (qos={:.2%})".format(costs.mean(), costs.std(),
                                                                   qos.mean()))

            print(f"{problem_type} dod{deg_of_dyna} test time(FT:{fine_tune_enable}): {time.time() - bt}s")
