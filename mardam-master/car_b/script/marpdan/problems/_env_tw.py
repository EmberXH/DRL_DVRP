from marpdan.problems import VRP_Environment
import torch

class VRPTW_Environment(VRP_Environment):
    CUST_FEAT_SIZE = 6

    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2, late_cost = 1):
        super().__init__(data, nodes, cust_mask, pending_cost)
        self.late_cost = late_cost

    def _sample_speed(self):
        return self.veh_speed

    def _update_vehicles(self, dest):
        dist = torch.pairwise_distance(self.cur_veh[:,0,:2], dest[:,0,:2], keepdim = True)
        tt = dist / self._sample_speed()   # 车辆到达下一个客户所需时间
        arv = torch.max(self.cur_veh[:,:,3] + tt, dest[:,:,3])  # 后者为左时间窗，self.cur_veh[:,:,3]猜测应该为0，不知为何考虑
        late = ( arv - dest[:,:,4] ).clamp_(min = 0)  # 超出右时间窗的时间，不能小于0

        self.cur_veh[:,:,:2] = dest[:,:,:2]  # 车辆坐标等于客户坐标
        self.cur_veh[:,:,2] -= dest[:,:,2]  # 车辆载重减去客户需求量
        self.cur_veh[:,:,3] = arv + dest[:,:,5]  # 车辆的next availability time为arv加服务时间

        self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE), self.cur_veh)
        return dist, late

    def step(self, cust_idx):
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        dist, late = self._update_vehicles(dest)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()
        reward = -dist - self.late_cost * late
        if self.done:
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            pending = (self.served ^ True).float().sum(-1, keepdim = True) - 1
            reward -= self.pending_cost * pending
        return reward
