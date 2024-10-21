import torch
from torch.utils.data import Dataset


class DVRPTW_QS_VB_Dataset(Dataset):
    CUST_FEAT_SIZE = 7

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask=None, changed_dem=None, b_veh_idx=None,
                 b_time=None):
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed

        self.nodes = nodes
        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.CUST_FEAT_SIZE, d))
        self.cust_mask = cust_mask
        self.changed_dem = changed_dem
        self.b_veh_idx = b_veh_idx
        self.b_time = b_time
        self.dod = 0.4
        self.d_early_ratio = 0.5

    @classmethod
    def generate(cls,
                 batch_size=1,
                 cust_count=100,
                 veh_count=25,
                 veh_capa=200,
                 veh_speed=1,
                 min_cust_count=None,
                 cust_loc_range=(0, 101),
                 cust_dem_range=(5, 41),
                 horizon=480,
                 cust_dur_range=(10, 31),
                 tw_ratio=0.5,
                 cust_tw_range=(30, 91),
                 dod=0.4,
                 d_early_ratio=0.5,
                 ):
        size = (batch_size, cust_count, 1)
        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count + 1, 2), dtype=torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype=torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durs = torch.randint(*cust_dur_range, size, dtype=torch.float)
        cls.dod = dod
        cls.d_early_ratio = d_early_ratio
        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)

        assert isinstance(dod, float)
        is_dyn_change = torch.empty(size).bernoulli_(dod)

        if isinstance(d_early_ratio, float):
            is_change_e = torch.empty(size).bernoulli_(d_early_ratio)
        elif len(d_early_ratio) == 1:
            is_change_e = torch.empty(size).bernoulli_(d_early_ratio[0])
        else:
            ratio = torch.tensor(d_early_ratio)[
                torch.randint(0, len(d_early_ratio), (batch_size,), dtype=torch.int64)]
            is_change_e = ratio[:, None, None].expand(*size).bernoulli()

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        b_veh_idx = torch.randint(low=0, high=int(veh_count), size=(batch_size, 1))
        b_time = torch.randint(low=horizon // 3 + 1, high=2 * horizon // 3 + 1, size=(batch_size, 1))

        change_time = is_dyn_change * is_change_e * torch.randint(1, horizon // 3 + 1, size, dtype=torch.float) \
                      + is_dyn_change * (1 - is_change_e) * torch.randint(horizon // 3 + 1, 2 * horizon // 3 + 1, size,
                                                                          dtype=torch.float)

        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else:  # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype=torch.int64)]
            has_tw = ratio[:, None, None].expand(*size).bernoulli()

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        # 时间窗宽度
        tws = (1 - has_tw) * torch.full(size, horizon) \
              + has_tw * torch.randint(*cust_tw_range, size, dtype=torch.float)

        tts = (locs[:, None, 0:1, :] - locs[:, 1:, None, :]).pow(2).sum(-1).pow(0.5) / veh_speed
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (change_time + torch.rand(size) * (horizon - torch.max(tts + durs, tws) - change_time))
        rdys.floor_()

        # Regroup all features in one tensor 坐标，需求，时间窗，服务时间，出现时间

        customers = torch.cat((locs[:, 1:], dems, rdys, rdys + tws, durs, change_time), 2)
        # 将在左时间窗后取消/更改的取消
        customers[:, :, 6] = torch.where((customers[:, :, 3] <= customers[:, :, 6]) & (has_tw.bool().squeeze()),
                                         0.0, customers[:, :, 6])
        # 定义扰动后结果范围
        min_perturbation = 0.5
        max_perturbation = 1.5
        input_tensor = customers[:, :, 2]
        # 定义整数扰动的范围
        integer_perturbation_range = 5  # 以5为例，可以根据实际需求修改
        integer_perturbation_matrix = torch.randint(-integer_perturbation_range, integer_perturbation_range + 1,
                                                    size=input_tensor.shape).float()
        # 扰动不能为0
        integer_perturbation_matrix = torch.where(integer_perturbation_matrix == 0,
                                                  torch.randint(1, integer_perturbation_range + 1,
                                                                size=integer_perturbation_matrix.shape).float(),
                                                  integer_perturbation_matrix)
        perturbed_tensor = input_tensor + integer_perturbation_matrix
        perturbed_tensor.clamp_(input_tensor * min_perturbation, input_tensor * max_perturbation)
        perturbed_tensor = perturbed_tensor.round()
        changed_dem = torch.where((customers[:, :, 6] > 0), perturbed_tensor,
                                  torch.zeros((batch_size, customers.size(1)), dtype=torch.float))

        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:, :, :2] = locs[:, 0:1]
        depot_node[:, :, 4] = horizon
        nodes = torch.cat((depot_node, customers), 1)
        # 加上仓库
        changed_dem = torch.cat((torch.zeros(batch_size, 1), changed_dem), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count + 1, cust_count + 2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(cust_count + 1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask, changed_dem, b_veh_idx, b_time)
        return dataset

    def normalize(self):
        loc_scl, loc_off = self.nodes[:, :, :2].max().item(), self.nodes[:, :, :2].min().item()
        loc_scl -= loc_off
        t_scl = self.nodes[:, 0, 4].max().item()

        self.nodes[:, :, :2] -= loc_off
        self.nodes[:, :, :2] /= loc_scl
        self.nodes[:, :, 2] /= self.veh_capa
        self.nodes[:, :, 3:] /= t_scl

        self.changed_dem /= self.veh_capa
        self.b_time = self.b_time.float() / t_scl

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m ^ 1] for n, m in zip(self.nodes, self.cust_mask))

    def save(self, fpath):
        torch.save({
            "veh_count": self.veh_count,
            "veh_capa": self.veh_capa,
            "veh_speed": self.veh_speed,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
        }, fpath)

    @classmethod
    def load(cls, fpath):
        return cls(**torch.load(fpath))

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.nodes[i], self.changed_dem[i], self.b_veh_idx[i], self.b_time[i]
        else:
            return self.nodes[i], self.cust_mask[i], self.changed_dem[i], self.b_veh_idx[i], self.b_time[i]

    def __len__(self):
        return self.batch_size
