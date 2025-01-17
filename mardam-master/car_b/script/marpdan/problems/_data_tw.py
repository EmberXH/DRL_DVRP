from marpdan.problems import VRP_Dataset
import torch


class VRPTW_Dataset(VRP_Dataset):
    CUST_FEAT_SIZE = 6

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
                 cust_tw_range=(30, 91)
                 ):
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count + 1, 2), dtype=torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype=torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durs = torch.randint(*cust_dur_range, size, dtype=torch.float)

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
        rdys = has_tw * (torch.rand(size) * (horizon - torch.max(tts + durs, tws)))
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locs[:, 1:], dems, rdys, rdys + tws, durs), 2)

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

        self.b_time = [i / t_scl for i in self.b_time]

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl
