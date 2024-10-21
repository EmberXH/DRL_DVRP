
import torch


class DVRPTW_QS_VB_Environment:
    CUST_FEAT_SIZE = 7
    VEH_STATE_SIZE = 4
    def __init__(self, data, nodes=None, cust_mask=None, changed_dem=None, b_idx=None, b_time=None,
                 pending_cost=2, late_cost=1,
                 speed_var=0.1, late_p=0.05, slow_down=0.5, late_var=0.3):

        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()

        self.pending_cost = pending_cost
        self.late_cost = late_cost
        self.speed_var = speed_var
        self.late_p = late_p
        self.slow_down = slow_down
        self.late_var = late_var

        # add
        self.cur_cust_idx = None
        self.is_ortools = False
        self.CUST_FEAT_SIZE = 7
        self.changed_dem = data.changed_dem if changed_dem is None else changed_dem
        self.b_veh_idx = data.b_veh_idx if b_idx is None else b_idx
        self.b_time = data.b_time if b_time is None else b_time
        self.b_veh_mask = torch.zeros((self.nodes.size(0), self.veh_count), dtype=torch.bool).to('cuda')

    def _update_hidden(self):  # 处理更改客户
        time = self.cur_veh[:, :, 3].clone()
        if self.init_cust_mask is None:
            change = ~self.served & (self.nodes[:, :, 6] <= time) & (
                    self.nodes[:, :, 6] != 0.0)  # 车辆时间大于取消时间，并且取消时间不能为0（不取消）
        else:
            change = ~(self.init_cust_mask | self.served) & (self.nodes[:, :, 6] < time) & (self.nodes[:, :, 6] != 0.0)
        if change.any():
            self.new_change = True
            new_nodes = self.nodes.clone()
            new_nodes[:, :, 2] = torch.where(change, self.changed_dem, new_nodes[:, :, 2])
            self.nodes = new_nodes
            # self.nodes_seq_cache.append(new_nodes)
            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            self._update_cur_veh()


    def reset(self):
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE)).to('cuda')
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]
        self.vehicles[:, :, 2] = self.veh_capa

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
        self.done = False

        self.cust_mask = torch.zeros((self.nodes.size(0), self.nodes.size(1)), dtype=torch.bool)
        if self.init_cust_mask is not None:
            self.cust_mask = self.cust_mask | self.init_cust_mask
        self.new_customers = True
        self.new_change = True
        self.served = torch.zeros_like(self.cust_mask).to('cuda')

        self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1).to('cuda')

        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype=torch.int64).to('cuda')
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE)).to(
            'cuda')
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count)).to(
            'cuda')
        # add
        self.veh_cur_cust = torch.zeros((self.minibatch_size, self.veh_count), dtype=torch.int64).to('cuda')

    def _sample_speed(self):
        # late = self.nodes.new_empty((self.minibatch_size, 1)).bernoulli_(self.late_p)
        # rand = torch.randn_like(late)
        # speed = late * self.slow_down * (1 + self.late_var * rand) + (1 - late) * (1 + self.speed_var * rand)
        # return speed.clamp_(min=0.1) * self.veh_speed
        return self.veh_speed

    def _update_vehicles(self, dest):
        dist = torch.pairwise_distance(self.cur_veh[:, 0, :2], dest[:, 0, :2], keepdim=True)
        tt = dist / self._sample_speed()  # 车辆到达下一个客户所需时间
        arv = torch.max(self.cur_veh[:, :, 3] + tt, dest[:, :, 3])  # 后者为左时间窗，self.cur_veh[:,:,3]猜测应该为0，不知为何考虑
        late = (arv - dest[:, :, 4]).clamp_(min=0)  # 超出右时间窗的时间，不能小于0

        self.cur_veh[:, :, :2] = dest[:, :, :2]  # 车辆坐标等于客户坐标
        self.cur_veh[:, :, 2] -= dest[:, :, 2]  # 车辆载重减去客户需求量
        self.cur_veh[:, :, 3] = arv + dest[:, :, 5]  # 车辆的next availability time为arv加服务时间

        self.vehicles = self.vehicles.scatter(1,
                                              self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
                                              self.cur_veh)
        return dist, late

    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())

    def _update_mask(self, cust_idx):
        self.change = False
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        overload = torch.zeros_like(self.mask).scatter_(1,
                                                        self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
                                                        self.cur_veh[:, :, None, 2] - self.nodes[:, None, :, 2] < 0)
        self.mask = self.mask | self.served[:, None, :] | overload | self.veh_done[:, :, None]
        self.mask[:, :, 0] = 0

    def _update_cur_veh(self):
        avail = self.vehicles[:, :, 3].clone()
        avail[self.veh_done] = float('inf')
        self.cur_veh_idx = avail.argmin(1, keepdim=True)
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))

    def step(self, cust_idx):
        new_veh_cur_cust = self.veh_cur_cust.clone()
        new_veh_cur_cust[torch.arange(self.veh_cur_cust.size(0)), self.cur_veh_idx.squeeze()] = cust_idx.squeeze()
        self.veh_cur_cust = new_veh_cur_cust
        self.cur_cust_idx = cust_idx
        dest = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
        dist, late = self._update_vehicles(dest)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        broken_happen = None
        if not self.is_ortools:
            broken_happen = (self.cur_veh_idx == self.b_veh_idx).squeeze() & (
                    torch.squeeze(self.cur_veh[:, :, 3]) > self.b_time[:, -1])
            all_tensor = torch.full(self.b_veh_mask.shape, True, dtype=torch.bool).to('cuda')
            a_expanded = broken_happen.view(-1, 1).expand(-1, self.b_veh_mask.size(1))
            b_expanded = self.b_veh_idx.expand(-1, self.b_veh_mask.size(1))
            idx_mask = (a_expanded & (b_expanded == torch.arange(self.b_veh_mask.size(1)).to('cuda')))
            self.b_veh_mask = torch.masked_scatter(self.b_veh_mask, idx_mask, all_tensor)
            self.veh_done = torch.masked_scatter(self.veh_done, idx_mask, all_tensor)

            all_tensor = torch.full(self.served.shape, False, dtype=torch.bool).to('cuda')
            a_expanded = broken_happen.view(-1, 1).expand(-1, self.served.size(1))
            b_expanded = self.cur_cust_idx.expand(-1, self.served.size(1))
            idx_mask = (a_expanded & (b_expanded == torch.arange(self.served.size(1)).to('cuda')))
            self.served = torch.masked_scatter(self.served, idx_mask, all_tensor)
            idx_mask = idx_mask.unsqueeze(1).expand(self.mask.size(0), self.mask.size(1), self.mask.size(2))
            all_tensor = torch.full(self.mask.shape, False, dtype=torch.bool).to('cuda')
            self.mask = torch.masked_scatter(self.mask, idx_mask, all_tensor)
            self.done = bool(self.veh_done.all())
            self.mask = self.mask | self.veh_done[:, :, None]
            self.mask[:, :, 0] = 0
        self._update_cur_veh()
        reward = -dist - self.late_cost * late
        if self.done:
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            pending = (self.served ^ True).float().sum(-1, keepdim=True) - 1
            reward -= self.pending_cost * pending
        # bh = self._update_broken()
        self._update_hidden()
        return reward, broken_happen

    def state_dict(self, dest_dict=None):
        if dest_dict is None:
            dest_dict = {
                "vehicles": self.vehicles,
                "veh_done": self.veh_done,
                "served": self.served,
                "mask": self.mask,
                "cur_veh_idx": self.cur_veh_idx
            }
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["veh_done"].copy_(self.veh_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
        return dest_dict

    def load_state_dict(self, state_dict):
        self.vehicles.copy_(state_dict["vehicles"])
        self.veh_done.copy_(state_dict["veh_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])

        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
