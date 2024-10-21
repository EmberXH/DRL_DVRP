from torch.utils.data import Dataset
import torch
import re


class DVRPTW_NR_Dataset(Dataset):
    CUST_FEAT_SIZE = 7

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask=None):
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed

        self.nodes = nodes
        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.CUST_FEAT_SIZE, d))
        self.cust_mask = cust_mask
        self.dod = 0.4
        self.d_early_ratio = 0.5

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.nodes[i]
        else:
            return self.nodes[i], self.cust_mask[i]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m ^ 1] for n, m in zip(self.nodes, self.cust_mask))

    @classmethod
    def read_instance(cls, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        dimension = capacity = None
        node_coords = {}
        demands = {}
        depot = None

        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD_SECTION"
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMAND_SECTION"
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT_SECTION"
            elif line.startswith("EOF"):
                break
            elif section == "NODE_COORD_SECTION":
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)
            elif section == "DEMAND_SECTION":
                parts = line.split()
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand

        batch_size = 1
        num_nodes = len(node_coords)
        size = (1, dimension - 1, 1)
        horizon = 480
        dod = 0.4
        d_early_ratio = 0.5

        match = re.search(r'k(\d+)', file_path)
        veh_count = int(match.group(1)) if match else 1
        veh_capa = capacity
        veh_speed = 1

        is_dyn = torch.empty(size).bernoulli_(dod)
        is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio)
        aprs = is_dyn * is_dyn_e * torch.randint(1, horizon // 3 + 1, size, dtype=torch.float) \
               + is_dyn * (1 - is_dyn_e) * torch.randint(horizon // 3 + 1, 2 * horizon // 3 + 1, size,
                                                         dtype=torch.float)

        customers = torch.zeros((batch_size, num_nodes - 1, cls.CUST_FEAT_SIZE - 1))
        depot = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot[:, :, 4] = horizon

        for i, (node_id, (x, y)) in enumerate(node_coords.items()):
            if i == 0:
                depot[0, 0, 0] = x
                depot[0, 0, 1] = y
                depot[0, 0, 2] = demands[node_id]
            else:
                customers[0, i - 1, 0] = x
                customers[0, i - 1, 1] = y
                customers[0, i - 1, 2] = demands[node_id]

        customers = torch.cat((customers, aprs), 2)
        nodes = torch.cat((depot, customers), 1)

        return cls(veh_count, veh_capa, veh_speed, nodes, None)

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
                 dod=0.5,
                 d_early_ratio=0.5
                 ):
        size = (batch_size, cust_count, 1)
        cls.dod = dod
        cls.d_early_ratio = d_early_ratio

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count + 1, 2), dtype=torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype=torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durs = torch.randint(*cust_dur_range, size, dtype=torch.float)

        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)
        if isinstance(dod, float):
            is_dyn = torch.empty(size).bernoulli_(dod)
        elif len(dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(dod[0])
        else:  # tuple of float
            ratio = torch.tensor(dod)[torch.randint(0, len(dod), (batch_size,), dtype=torch.int64)]
            is_dyn = ratio[:, None, None].expand(*size).bernoulli()

        if isinstance(d_early_ratio, float):
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio)
        elif len(d_early_ratio) == 1:
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio[0])
        else:
            ratio = torch.tensor(d_early_ratio)[
                torch.randint(0, len(d_early_ratio), (batch_size,), dtype=torch.int64)
            ]
            is_dyn_e = ratio[:, None, None].expand(*size).bernoulli()

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        aprs = is_dyn * is_dyn_e * torch.randint(1, horizon // 3 + 1, size, dtype=torch.float) \
               + is_dyn * (1 - is_dyn_e) * torch.randint(horizon // 3 + 1, 2 * horizon // 3 + 1, size,
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
        tws = (1 - has_tw) * torch.full(size, horizon) \
              + has_tw * torch.randint(*cust_tw_range, size, dtype=torch.float)

        tts = (locs[:, None, 0:1, :] - locs[:, 1:, None, :]).pow(2).sum(-1).pow(0.5) / veh_speed
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (aprs + torch.rand(size) * (horizon - torch.max(tts + durs, tws) - aprs))
        rdys.floor_()

        # Regroup all features in one tensor 坐标，需求，时间窗，服务时间，出现时间
        customers = torch.cat((locs[:, 1:], dems, rdys, rdys + tws, durs, aprs), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:, :, :2] = locs[:, 0:1]
        depot_node[:, :, 4] = horizon
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count + 1, cust_count + 2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(cust_count + 1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset

    def normalize(self):
        loc_scl, loc_off = self.nodes[:, :, :2].max().item(), self.nodes[:, :, :2].min().item()
        loc_scl -= loc_off
        t_scl = self.nodes[:, 0, 4].max().item()

        self.nodes[:, :, :2] -= loc_off
        self.nodes[:, :, :2] /= loc_scl
        self.nodes[:, :, 2] /= self.veh_capa
        self.nodes[:, :, 3:] /= t_scl

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl

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
