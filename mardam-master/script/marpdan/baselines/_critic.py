from marpdan.baselines._base import Baseline
import sys

sys.path.append("..")
from marpdan.problems import *
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
import dgl
import scipy.sparse as sp
import time


class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval=True, use_cumul_reward=False):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        self.project = nn.Linear(cust_count + 1, cust_count + 1 if use_qval else 1, bias=False)

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        compat = learner_compat.clone()
        compat[vrp_dynamics.cur_veh_mask] = 0
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1, 1, -1))
        return val.squeeze(1)

    def __call__(self, vrp_dynamics):
        vrp_dynamics.reset()
        actions, logps, rewards, bl_vals = [], [], [], []
        node_features = vrp_dynamics.nodes[-1, vrp_dynamics.CUST_FEAT_SIZE]
        if type(vrp_dynamics) == DVRPTW_TJ_Environment or type(vrp_dynamics) == DVRPTW_QC_Environment or type(
                vrp_dynamics) == DVRPTW_NR_Environment or type(vrp_dynamics) == DVRPTW_QS_Environment or type(
                vrp_dynamics) == DVRPTW_VB_Environment or type(vrp_dynamics) == DVRPTW_NR_TJ_Environment or type(
                vrp_dynamics) == DVRPTW_QS_VB_Environment:
            # QC encode_customers待更改, NR 原版没有动态更新encode_customers
            # self.learner._encode_customers(vrp_dynamics.nodes)
            self.learner._encode_customers(vrp_dynamics.nodes,
                                           torch.full((vrp_dynamics.minibatch_size, vrp_dynamics.nodes_count-1), 0).to('cuda'))
            while not vrp_dynamics.done:
                # veh_repr = self.learner._repr_vehicle(
                #     vrp_dynamics.vehicles,
                #     vrp_dynamics.cur_veh_idx,
                #     vrp_dynamics.mask)
                veh_repr = self.learner._decode(vrp_dynamics.veh_cur_cust,
                                                vrp_dynamics.vehicles,
                                                vrp_dynamics.cur_veh_idx,
                                                vrp_dynamics.mask)
                compat = self.learner._score_customers(veh_repr)
                logp = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
                cust_idx = logp.exp().multinomial(1)
                if not (self.use_cumul and bl_vals):
                    bl_vals.append(self.eval_step(vrp_dynamics, compat, cust_idx))
                ## actions.append(torch.cat((vrp_dynamics.cur_veh_idx, cust_idx), dim=1))
                actions.append((vrp_dynamics.cur_veh_idx, cust_idx))
                logps.append(logp.gather(1, cust_idx))
                r = vrp_dynamics.step(cust_idx)
                rewards.append(r)
                # 也可以不置0
                # if bh.any():
                #     for i in range(vrp_dynamics.minibatch_size):
                #         if bh[i]:
                #             bl_vals[-1][i, :] = 0
                #             logps[-1][i, :].zero_()
                #             rewards[-1][i, :] = 0
        else:
            raise NotImplementedError
        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim=0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device=device)

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = (
            adj_.dot(degree_mat_inv_sqrt)
            .transpose()
            .dot(degree_mat_inv_sqrt)
            .tocoo()
        )
        return adj_normalized

    def preprocess_graph_plus(self, adj):
        # Convert adjacency matrix to a list of lists
        adj_list = [list(row) for row in adj]
        # Add self-loops
        for i in range(len(adj_list)):
            adj_list[i][i] += 1
        # Calculate row sums
        rowsum = [sum(row) for row in adj_list]
        # Calculate degree matrix with inverse square root
        degree_mat_inv_sqrt = [[1.0 / (rowsum[i] ** 0.5) if rowsum[i] != 0 else 0.0 for i in range(len(adj_list))] for _
                               in range(len(adj_list))]
        # Normalize adjacency matrix
        adj_normalized = [
            [sum(adj_list[j][i] * degree_mat_inv_sqrt[i][i] * degree_mat_inv_sqrt[j][j] for j in range(len(adj_list)))
             for i in range(len(adj_list))] for _ in range(len(adj_list))]
        return adj_normalized
