from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import eval_apriori_routes

import torch
from torch.utils.data import DataLoader
import pickle
import os

BATCH_SIZE = 10000  # 10000
SEED = 231034871114
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]
ROLLOUTS = 100

torch.manual_seed(SEED)

problem_type = "nr_tj"
Dataset = {
    "tj": DVRPTW_TJ_Dataset,
    "qs": DVRPTW_QS_Dataset,
    "qc": DVRPTW_QC_Dataset,
    "vb": DVRPTW_VB_Dataset,
    "nr": DVRPTW_NR_Dataset,
    "nr_tj": DVRPTW_NR_TJ_Dataset,
    "qs_vb": DVRPTW_QS_VB_Dataset
}.get(problem_type)

# for n, m in ((10, 2), (20, 4), (50, 10)):
n = 50
m = n // 5
dods = [0.3,0.4,0.6]
# ort_routes = ort_solve(data)
for dod in dods:
    data = Dataset.generate(BATCH_SIZE, n, m, dod=dod)
    data.normalize()
    out_dir = "data/{}_n{}m{}_{}_{}_{}".format(problem_type, n, m, dod, Dataset.d_early_ratio, BATCH_SIZE)
    os.makedirs(out_dir, exist_ok=True)

    # env = VRPTW_Environment(data)
    # ort_costs = eval_apriori_routes(env, ort_routes, 1)

    os.makedirs(os.path.dirname(os.path.join(out_dir, "norm_data.pyth")), exist_ok=True)
    torch.save(data, os.path.join(out_dir, "norm_data.pyth"))
    print("Test data generation completed! Saved in {}".format(out_dir))
    # os.makedirs(os.path.dirname(os.path.join(out_dir, "ort.pyth")), exist_ok=True)
    # torch.save({
    #     "costs": ort_costs,
    #     "routes": ort_routes,
    #     }, os.path.join(out_dir, "ort.pyth"))
