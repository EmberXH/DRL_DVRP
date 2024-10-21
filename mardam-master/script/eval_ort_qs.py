import copy

from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm

from argparse import ArgumentParser
import os.path
from multiprocessing import Pool
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from marpdan.problems import *
from marpdan.utils import eval_apriori_routes
import time

# ORTOOLS
DEPOT = 0
LOCX, LOCY, DEM, RDY, DUE, DUR, CHA = range(7)
LOC = slice(0, 2)
TIMES = slice(3, 7)


def _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial_routes=None, t=0):
    horizon = int(nodes[DEPOT, DUE])
    nodes_count = nodes.size(0)
    # vis_a = nodes[:, CHA] > t
    # vis_b = nodes[:, CHA] == 0.0
    # vis = vis_a | vis_b
    vis = torch.ones(nodes_count).bool()
    # if partial_routes is not None:
    #     flat_list = [item for sublist in partial_routes for item in sublist]
    #     vis[flat_list] = True

    vis_idx = torch.arange(nodes_count)[vis].tolist()
    vis_count = int(vis.sum())

    manager = pywrapcp.RoutingIndexManager(vis_count, veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_i, to_i):
        from_n = manager.IndexToNode(from_i)
        to_n = manager.IndexToNode(to_i)
        return int(dist[vis_idx[from_n], vis_idx[to_n]])

    dist_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    def dem_cb(from_i):
        from_n = manager.IndexToNode(from_i)
        return int(nodes[vis_idx[from_n], DEM])

    dem_cb_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimension(dem_cb_idx, 0, veh_capa, True, "Capacity")

    def t_cb(from_i, to_i):
        from_n = manager.IndexToNode(from_i)
        to_n = manager.IndexToNode(to_i)
        return int(nodes[vis_idx[from_n], DUR] + dist[vis_idx[from_n], vis_idx[to_n]] / veh_speed)

    t_cb_idx = routing.RegisterTransitCallback(t_cb)
    routing.AddDimension(t_cb_idx, horizon, 2 * horizon, False, "Time")

    t_dim = routing.GetDimensionOrDie("Time")
    for from_n, vis_i in enumerate(vis_idx[1:]):
        from_i = manager.NodeToIndex(from_n)
        t_dim.CumulVar(from_i).SetMin(int(nodes[vis_i, RDY]))
        t_dim.SetCumulVarSoftUpperBound(from_i, int(nodes[vis_i, DUE]), late_cost)
    for veh_i in range(veh_count):
        end_i = routing.End(veh_i)
        t_dim.SetCumulVarSoftUpperBound(end_i, horizon, late_cost)
        routing.AddVariableMinimizedByFinalizer(t_dim.CumulVar(end_i))

    for to_n in range(1, vis_count):
        routing.AddDisjunction([manager.NodeToIndex(to_n)], pending_cost, 1)

    if partial_routes is not None:
        routing.CloseModel()
        rev_idx = {vis_i: to_n for to_n, vis_i in enumerate(vis_idx)}
        locks = [[rev_idx[vis_i] for vis_i in route] for route in partial_routes]
        routing.ApplyLocksToAllVehicles(locks, False)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.solution_limit = 20
    solution = routing.SolveWithParameters(params)
    if not solution:
        return None, None
    routes = [[] for _ in range(veh_count)]
    traj = []
    for veh_i in range(veh_count):
        t = 0
        to_i = routing.Start(veh_i)
        while not routing.IsEnd(to_i):
            to_i = solution.Value(routing.NextVar(to_i))
            to_n = manager.IndexToNode(to_i)
            routes[veh_i].append(vis_idx[to_n])
            traj.append((t, veh_i, vis_idx[to_n]))
            t = solution.Value(t_dim.CumulVar(to_i)) + int(nodes[vis_idx[to_n], DUR])
    traj.sort()
    return routes, traj


def _solve_loop(nodes, veh_count, veh_capa, veh_speed, pending_cost, late_cost, changed_dem):
    hidden_changed = nodes[:, CHA] > 0
    changed = []
    for i in range(len(hidden_changed)):
        if hidden_changed[i]:
            changed.append((i, nodes[i, CHA]))

    changed = sorted(changed, key=lambda x: x[1], reverse=True)

    dist = (nodes[:, None, LOC] - nodes[None, :, LOC]).pow(2).sum(axis=2).pow(0.5).ceil()

    partial = None
    t = 0
    while changed:
        routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost,
                                 partial, t)
        if not routes:
            return None
        partial = [[] for _ in range(veh_count)]
        for t, veh_i, to_n in traj:
            if t < changed[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while changed and t >= changed[-1][1]:
                c = changed.pop()
                nodes[c[0], 2] = changed_dem[c[0]]
            break
        else:
            c = changed.pop()
            t = c[1]
            nodes[c[0], 2] = changed_dem[c[0]]
    routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial, t)
    return routes


def ort_solve_dyna(data, pending_cost=200, late_cost=1, no_mp=False):
    if no_mp:
        routes = []
        for pr_data in tqdm(data):
            routes.append(_solve_loop(pr_data[0], data.veh_count, data.veh_capa, data.veh_speed,
                                      pending_cost, late_cost, pr_data[1]))
        return routes
    else:
        with Pool() as p:
            with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
                results = [p.apply_async(_solve_loop, (pr_data[0], data.veh_count, data.veh_capa, data.veh_speed,
                                                       pending_cost, late_cost, pr_data[1]),
                                         callback=lambda _: pbar.update()) for pr_data in data]
                routes = [res.get(timeout=2400) for res in results]
    return routes


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument("data_path")
    parser.add_argument("--normalized-data", "-n", default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pending-cost", type=int, default=200)
    parser.add_argument("--late-cost", type=int, default=1)
    parser.add_argument("--no-mp", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    args.no_mp = True
    problem_size = 100
    deg_of_dynas = [0.3, 0.4, 0.6]
    appear_early_ratio = 0.5
    for deg_of_dyna in deg_of_dynas:
        data_path = "./data/qs_n{}m{}_{}_{}_10000/norm_data.pyth".format(problem_size, problem_size // 5, deg_of_dyna,
                                                                         appear_early_ratio)
        data = torch.load(data_path)
        assert isinstance(data, DVRPTW_QS_Dataset)

        if args.normalized_data:
            unnorm = data.nodes.clone()
            unnorm[:, :, LOC] *= 100
            unnorm[:, :, DEM] *= 200
            unnorm[:, :, TIMES] *= 480
            unnorm_changed_dem = data.changed_dem.clone()
            unnorm_changed_dem *= 200
            unnorm_data = DVRPTW_QS_Dataset(data.veh_count, 200, 1, unnorm, data.cust_mask, unnorm_changed_dem)
        else:
            nodes = data.nodes.clone()
            unnorm_data = DVRPTW_QS_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes, data.cust_mask,
                                            data.changed_dem)
            data.normalize()

        bt = time.time()
        routes = ort_solve_dyna(unnorm_data, args.pending_cost, args.late_cost, args.no_mp)
        torch.save(routes, "DUMP_routes_dyn.pyth")
        # routes = torch.load("DUMP_routes_dyn.pyth")
        routes_len = len(routes)
        # 使用列表推导式获取非 None 的元素和其索引
        final_routes = [element for index, element in enumerate(routes) if element is not None]
        # 使用列表推导式获取为 None 的元素的索引
        none_indices = [index for index, element in enumerate(routes) if element is None]
        data.batch_size = len(final_routes)
        for i in range(len(none_indices)):
            data.nodes = torch.cat((data.nodes[:none_indices[i]], data.nodes[none_indices[i] + 1:]), dim=0)
            data.changed_dem = torch.cat((data.changed_dem[:none_indices[i]], data.changed_dem[none_indices[i] + 1:]),
                                         dim=0)
        print("solve rate: {}".format((routes_len - len(none_indices)) / routes_len))
        loader = DataLoader(data, batch_size=50)  # 50
        with tqdm(loader, "Evaluating") as progress:
            costs = []
            qos = []
            for b, batch in enumerate(progress):
                assert len(batch) == 2
                env = DVRPTW_QS_Environment(data, batch[0], None, batch[1], pending_cost=0, late_p=0)
                env.nodes = env.nodes.to("cuda")
                env.changed_dem = env.changed_dem.to("cuda")
                costs.append(eval_apriori_routes(env, final_routes[50 * b:50 * (b + 1)], 1))
                pending = (env.served ^ True).float().sum(-1) - 1
                qos.append(1 - pending / (env.nodes_count - 1))
            costs = torch.cat(costs, 0)
            qos = torch.cat(qos, 0)

        if args.output_dir is None:
            args.output_dir = os.path.dirname(data_path).replace("data", "results")

        print("{:5.2f} +- {:5.2f} (qos={:.2%})".format(costs.mean(), costs.std(), qos.mean()))

        print(f"total test time: {time.time() - bt}")


if __name__ == "__main__":
    main(parse_args())
