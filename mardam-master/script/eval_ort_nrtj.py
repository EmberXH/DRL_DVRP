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
LOCX, LOCY, DEM, RDY, DUE, DUR, APR = range(7)
LOC = slice(0, 2)
TIMES = slice(3, 7)


def _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial_routes=None, t=0):
    horizon = int(nodes[DEPOT, DUE])
    nodes_count = nodes.size(0)

    vis =nodes[:, APR] <= t
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


def _solve_loop(nodes, veh_count, veh_capa, veh_speed, pending_cost, late_cost, tj_time, distance_with_tj):
    # tjs = [[i.item(), j.item(), tj_time[i, j].item()] for i, j in torch.nonzero(tj_time)]
    # tjs = sorted(tjs, key=lambda x: x[2], reverse=True)
    hidden = nodes[:, APR] > 0
    aprs = nodes[hidden, APR].tolist()
    aprs = [[0, i] for i in aprs]

    tjts = [[1, tj_time[i, j].item()] for i, j in torch.nonzero(tj_time)]
    event_times = sorted(aprs + tjts, key=lambda x: x[-1], reverse=True)
    dist = (nodes[:, None, LOC] - nodes[None, :, LOC]).pow(2).sum(axis=2).pow(0.5).ceil()

    partial = None
    t = 0
    while event_times:
        routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost,
                                 partial, t)
        partial = [[] for _ in range(veh_count)]
        for t, veh_i, to_n in traj:
            if t < event_times[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while event_times and t >= event_times[-1][1]:
                et = event_times.pop()
                if et[0] == 1:
                    rows, cols = torch.where(tj_time == et[1])
                    for row, col in zip(rows, cols):
                        dist[row, col] = distance_with_tj[row, col]
            break
        else:
            et = event_times.pop()
            t = et[1]
            if et[0] == 1:
                rows, cols = torch.where(tj_time == et[1])
                for row, col in zip(rows, cols):
                    dist[row, col] = distance_with_tj[row, col]
    routes, traj = _solve_cp(nodes, dist, veh_count, veh_capa, veh_speed, pending_cost, late_cost, partial, t)
    return routes


def ort_solve_dyna(data, pending_cost=200, late_cost=1, no_mp=False):
    if no_mp:
        routes = []
        for pr_data in tqdm(data):
            routes.append(_solve_loop(pr_data[0], data.veh_count, data.veh_capa, data.veh_speed,
                                      pending_cost, late_cost, pr_data[1], pr_data[2]))
        return routes
    else:
        with Pool() as p:
            with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
                results = [p.apply_async(_solve_loop, (pr_data[0], data.veh_count, data.veh_capa, data.veh_speed,
                                                       pending_cost, late_cost, pr_data[1], pr_data[2]),
                                         callback=lambda _: pbar.update()) for pr_data in data]
                routes = [res.get(timeout=240) for res in results]
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
    deg_of_dynas = [0.3,0.4,0.6]
    appear_early_ratio = 0.5
    for deg_of_dyna in deg_of_dynas:
        data_path = "./data/nr_tj_n{}m{}_{}_{}_10000/norm_data.pyth".format(problem_size, problem_size // 5, deg_of_dyna,
                                                                         appear_early_ratio)
        data = torch.load(data_path)
        assert isinstance(data, DVRPTW_NR_TJ_Dataset)
        if args.normalized_data:
            unnorm = data.nodes.clone()
            unnorm[:, :, LOC] *= 100
            unnorm[:, :, DEM] *= 200
            unnorm[:, :, TIMES] *= 480
            unnorm_tj_time = data.tj_time.clone()
            unnorm_tj_time *= 480
            unnorm_dwt = data.distances_with_tj.clone()
            unnorm_dwt *= 100
            unnorm_data = DVRPTW_NR_TJ_Dataset(data.veh_count, 200, 1, unnorm, data.cust_mask, unnorm_tj_time, unnorm_dwt)
        else:
            nodes = data.nodes.clone()
            unnorm_data = DVRPTW_NR_TJ_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes, data.cust_mask,
                                               data.changed_locs)
            data.normalize()

        bt = time.time()
        routes = ort_solve_dyna(unnorm_data, args.pending_cost, args.late_cost, args.no_mp)
        torch.save(routes, "DUMP_routes_dyn.pyth")
        # routes = torch.load("DUMP_routes_dyn.pyth")

        loader = DataLoader(data, batch_size=50)  # 50
        data.nodes = data.nodes.to('cuda')
        data.tj_time = data.tj_time.to('cuda')
        data.distances_with_tj = data.distances_with_tj.to('cuda')
        with tqdm(loader, "Evaluating") as progress:
            costs = []
            qos = []
            for b, batch in enumerate(progress):
                env = DVRPTW_NR_TJ_Environment(data, batch[0], None, batch[1], batch[2], pending_cost=0, late_p=0)
                env.nodes = env.nodes.to("cuda")
                costs.append(eval_apriori_routes(env, routes[50 * b:50 * (b + 1)], 1))
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
