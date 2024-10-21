from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = False
NO_CUDA = False

SEED = 1234567890   # 重要3333
PROBLEM = "nr"    # 用于计算ref_cost
CUST_COUNT = 20  # gene_p
VEH_COUNT = CUST_COUNT // 5    # gene_p
VEH_CAPA = 200  # gene_p
VEH_SPEED = 1  # gene_p
HORIZON = 480  # gene_p
MIN_CUST_COUNT = None # gene_p
LOC_RANGE = (0, 101)  # gene_p
DEM_RANGE = (5, 41)   # gene_p
DUR_RANGE = (10, 31)  # gene_p
TW_RATIO = (0.25, 0.5, 0.75, 1.0)  # gene_p
TW_RANGE = (30, 91)  # gene_p
DEG_OF_DYN = 0.4  # gene_p
APPEAR_EARLY_RATIO = (0.0, 0.5, 0.75, 1.0)  # gene_p

PEND_COST = 2  # env_p
PEND_GROWTH = None
LATE_COST = 1  # env_p
LATE_GROWTH = None
SPEED_VAR = 0.1  # env_p
LATE_PROB = 0.05  # env_p
SLOW_DOWN = 0.5  # env_p
LATE_VAR = 0.2  # env_p

MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8   # 8
FF_SIZE = 512
TANH_XPLOR = 10

EPOCH_COUNT = 40000    # meta 10000 # nometa 20b
ITER_COUNT = 1000
MINIBATCH_SIZE = 1024     # n50:1024 n100:128
BASE_LR = 0.0001
LR_DECAY = None
TW_R = 0.5
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = False   # baseline

BASELINE = "critic"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = False   # baseline
CRITIC_LR = 0.001
CRITIC_DECAY = 0.0001

TEST_BATCH_SIZE = 128

OUTPUT_DIR = None
RESUME_STATE = None
CHECKPOINT_PERIOD = 1000   # 存储ckp的间隔


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)


def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-f", type=str, default=CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action="store_true", default=VERBOSE)
    parser.add_argument("--no-cuda", action="store_true", default=NO_CUDA)
    parser.add_argument("--rng-seed", type=int, default=SEED)

    group = parser.add_argument_group("Data generation parameters")
    group.add_argument("--problem-type", "-p", type=str, default=PROBLEM)
    group.add_argument("--customers-count", "-n", type=int, default=CUST_COUNT)
    group.add_argument("--vehicles-count", "-m", type=int, default=VEH_COUNT)
    group.add_argument("--veh-capa", type=int, default=VEH_CAPA)
    group.add_argument("--veh-speed", type=int, default=VEH_SPEED)
    group.add_argument("--horizon", type=int, default=HORIZON)
    group.add_argument("--min-cust-count", type=int, default=MIN_CUST_COUNT)
    group.add_argument("--loc-range", type=int, nargs=2, default=LOC_RANGE)
    group.add_argument("--dem-range", type=int, nargs=2, default=DEM_RANGE)
    group.add_argument("--dur-range", type=int, nargs=2, default=DUR_RANGE)
    group.add_argument("--tw-ratio", type=float, nargs='*', default=TW_RATIO)
    group.add_argument("--tw-range", type=int, nargs=2, default=TW_RANGE)
    group.add_argument("--deg-of-dyna", type=float, nargs='*', default=DEG_OF_DYN)
    group.add_argument("--appear-early-ratio", type=float, nargs='*', default=APPEAR_EARLY_RATIO)

    group = parser.add_argument_group("VRP Environment parameters")
    group.add_argument("--pending-cost", type=float, default=PEND_COST)
    group.add_argument("--pend-cost-growth", type=float, default=PEND_GROWTH)
    group.add_argument("--late-cost", type=float, default=LATE_COST)
    group.add_argument("--late-cost-growth", type=float, default=LATE_GROWTH)
    group.add_argument("--speed-var", type=float, default=SPEED_VAR)
    group.add_argument("--late-prob", type=float, default=LATE_PROB)
    group.add_argument("--slow-down", type=float, default=SLOW_DOWN)
    group.add_argument("--late-var", type=float, default=LATE_VAR)
    group.add_argument("--tw_r", default=TW_R)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model-size", "-s", type=int, default=MODEL_SIZE)
    group.add_argument("--layer-count", type=int, default=LAYER_COUNT)
    group.add_argument("--head-count", type=int, default=HEAD_COUNT)
    group.add_argument("--ff-size", type=int, default=FF_SIZE)
    group.add_argument("--tanh-xplor", type=float, default=TANH_XPLOR)

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epoch-count", "-e", type=int, default=EPOCH_COUNT)
    group.add_argument("--iter-count", "-i", type=int, default=ITER_COUNT)
    group.add_argument("--batch-size", "-b", type=int, default=MINIBATCH_SIZE)
    group.add_argument("--learning-rate", "-r", type=float, default=BASE_LR)
    group.add_argument("--rate-decay", "-d", type=float, default=LR_DECAY)
    group.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    group.add_argument("--grad-norm-decay", type=float, default=GRAD_NORM_DECAY)
    group.add_argument("--loss-use-cumul", action="store_true", default=LOSS_USE_CUMUL)

    group = parser.add_argument_group("Baselines parameters")
    group.add_argument("--baseline-type", type=str,
                       choices=["none", "nearnb", "rollout", "critic"], default=BASELINE)
    group.add_argument("--rollout-count", type=int, default=ROLLOUT_COUNT)
    group.add_argument("--rollout-threshold", type=float, default=ROLLOUT_THRESHOLD)
    group.add_argument("--critic-use-qval", action="store_true", default=CRITIC_USE_QVAL)
    group.add_argument("--critic-rate", type=float, default=CRITIC_LR)
    group.add_argument("--critic-decay", type=float, default=CRITIC_DECAY)

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--test-batch-size", type=int, default=TEST_BATCH_SIZE)

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output-dir", "-o", type=str, default=OUTPUT_DIR)
    group.add_argument("--checkpoint-period", "-c", type=int, default=CHECKPOINT_PERIOD)
    group.add_argument("--resume-state", type=str, default=RESUME_STATE)

    group = parser.add_argument_group("Meta params")
    group.add_argument("--meta_enable", default=False)
    group.add_argument("--curriculum", default=True)
    group.add_argument("--meta_method", default="fomaml")
    group.add_argument("--k", default=1)
    group.add_argument("--B", default=1)
    group.add_argument("--update_interval", default=100)
    group.add_argument("--alpha", default=1)  # only for reptile
    group.add_argument("--alpha_decay", default=0.99)
    group.add_argument("--max_dod", default=0.2)
    group.add_argument("--min_dod", default=0.0)
    group.add_argument("--dod_interval", default=0.05)
    group.add_argument("--problem_type_num", default=3)
    group.add_argument("--inner-learning-rate", type=float, default=0.001)
    group.add_argument("--inner-rate-decay", type=float, default=0.0001)
    group.add_argument("--ref-enable", default=False)   # 实时使用验证集进行验证，开启增大训练时间，下面是验证的间隔
    group.add_argument("--val-interval", default=5)

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args(argv)
    if args.meta_enable:
        args.tw_r = args.tw_ratio
    return args
