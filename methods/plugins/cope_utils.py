from collections import defaultdict

import torch


class PrototypeScheme(object):
    valid_p_modes = ['batch_only', 'batch_momentum', 'batch_momentum_it',
                     'batch_momentum_incr']

    def __init__(self, p_mode='batch_momentum_incr', p_momentum=0.9):
        """
        :param p_mode: prototypes update mode
        """
        assert p_mode in self.valid_p_modes, "{} not in {}".format(p_mode,
                                                                   self.valid_p_modes)

        self.p_momentum = p_momentum
        self.update_pre_loss = False
        self.update_post_loss = True

        self.device = torch.device

        if p_mode == 'batch_only':
            self.p_update = self.update_batch_momentum
            self.p_momentum = 0  # Old value is zero, only current batch counts
        elif p_mode == 'batch_momentum':
            self.p_update = self.update_batch_momentum
        elif p_mode == 'batch_momentum_it':
            self.p_update = self.update_batch_momentum
            self.update_pre_loss = True  # Update each iteration
            self.update_post_loss = False
        elif p_mode == 'batch_momentum_incr':
            self.p_update = self.update_batch_momentum_incr
            self.update_pre_loss = True  # Accumulate batch info
            self.update_post_loss = True  # -> Only actual update
        else:
            raise NotImplementedError()

        self.prototypes = {}
        self.class_mem = {}

        self.p_tmp_cnt = defaultdict(int)
        self.p_tmp = {}

    def add_prototypes(self, classes, f):
        for c in set(classes):
            self.prototypes[c] = self.init_prototype_val(f)

    def get_prototypes(self):
        p = torch.cat(list(self.prototypes.values()), 0)
        y = torch.tensor(list(self.prototypes.keys()), device=p.device)

        return p, y

    def initialize_prototypes(self, o, classes):
        for c in classes:
            if c not in self.prototypes:
                p = torch.empty((1, o.shape[-1]), device=o.device).uniform_(0, 1)
                p = torch.nn.functional.normalize(p, p=2, dim=1).detach()

                self.prototypes[c] = p

    @staticmethod
    def init_prototype_val(o):
        p = torch.empty((1, o.shape[-1]), device=o.device).uniform_(0, 1)
        p = torch.nn.functional.normalize(p, p=2, dim=1).detach()
        return p

    @torch.no_grad()
    def __call__(self, f, y, replay_mask, pre_loss=False):
        if (pre_loss and self.update_pre_loss) or \
                (not pre_loss and self.update_post_loss):

            self.p_update(f=f, y=y,
                          class_mem=None, replay_mask=replay_mask, pre_loss=pre_loss)

    @staticmethod
    def momentum_update(old_value, new_value, momentum):
        update = momentum * old_value + (1 - momentum) * new_value
        return update

    def update_batch_momentum_incr(self, f, y, replay_mask,
                                   pre_loss, **kwargs):
        device = f.device

        y_new, f_new = y[~replay_mask], f[~replay_mask]
        y_memory, f_memory = y[replay_mask], f[replay_mask]

        if pre_loss and len(y_memory) > 0:  # Accumulate replayed exemplars (/class)
            unique_labels = torch.unique(y_memory)
            for label_idx in range(unique_labels.size(0)):
                c = unique_labels[label_idx]
                idxs = (y_memory == c).nonzero().squeeze(1)

                p_tmp_batch = f_memory[idxs].sum(dim=0).unsqueeze(0)
                p_tmp_batch = p_tmp_batch.to(device)

                self.p_tmp[c] = self.p_tmp.get(c, torch.zeros_like(
                    p_tmp_batch)) + p_tmp_batch
                self.p_tmp_cnt[c.item()] += len(idxs)
        else:
            for c, _ in self.prototypes.items():
                # Include new ones too (All replayed already in pre-loss)
                idxs_new = (y_new == c).nonzero().squeeze(1)

                if len(idxs_new) > 0:
                    p_tmp_batch = f_new[idxs_new].sum(dim=0).unsqueeze(0)
                    p_tmp_batch = p_tmp_batch.to(device)

                    # print(p_tmp_batch.shape, self.p_tmp.get(c, torch.zeros_like(p_tmp_batch)).shape)
                    self.p_tmp[c] = self.p_tmp.get(c, torch.zeros_like(p_tmp_batch)) + p_tmp_batch
                    self.p_tmp_cnt[c] += len(idxs_new)

                    incr_p = self.p_tmp[c] / self.p_tmp_cnt[c]

                    old_p = self.prototypes[c].clone()
                    new_p = self.momentum_update(old_p, incr_p, self.p_momentum)
                    # Uncomment to renormalize prototypes:
                    # new_p_momentum = self.momentum_update(old_p, incr_p, self.p_momentum)
                    # new_p = torch.nn.functional.normalize(new_p_momentum, p=2,dim=1).detach()

                    # Update
                    self.prototypes[c] = new_p
                    assert not torch.isnan(self.prototypes[c].any())

                    # Re-init
                    del self.p_tmp[c]
                    self.p_tmp_cnt[c] = 0

    def update_batch_momentum(self, f, y, **kwargs):
        """Take momentum of current batch avg with prev prototype (based on prev batches)."""
        unique_labels = torch.unique(y).squeeze()

        for label_idx in range(unique_labels.size(0)):
            c = unique_labels[label_idx]
            idxs = (y == c).nonzero().squeeze(1)

            batch_p = f[idxs].mean(0).unsqueeze(0)  # Mean of whole batch
            # old_p = class_mem[c.item()].prototype
            old_p = self.prototypes.get(c.item(), self.init_prototype_val(f))

            new_p = self.momentum_update(old_p, batch_p, self.p_momentum)

            self.prototypes[c.item()] = new_p


class PPPloss(object):
    """ Pseudo-Prototypical Proxy loss
    """
    modes = ["joint", "pos", "neg"]  # Include attractor (pos) or repellor (neg) terms

    def __init__(self, mode="joint", T=1, tracker=None, ):
        """
        :param margin: margin on distance between pos vs neg samples (see TripletMarginLoss)
        :param dist: distance function 2 vectors (e.g. L2-norm, CosineSimilarity,...)
        """
        assert mode in self.modes
        self.mode = mode
        self.T = T
        self.margin = 1

    def __call__(self, x_metric, labels, prototypes, eps=1e-8):
        """
        Standard reduction is mean, as we use full batch information instead of per-sample.
        Symmetry in the distance function inhibits summing loss over all samples.
        :param x_metric: embedding output of batch
        :param labels: labels batch
        :param class_mem: Stored prototypes/exemplars per seen class
        """
        if self.mode == "joint":
            pos, neg = True, True
        elif self.mode == "pos":
            pos, neg = True, False
        elif self.mode == "neg":
            pos, neg = False, True
        else:
            raise NotImplementedError()
        return self.softmax_joint(x_metric, labels, prototypes, pos=pos, neg=neg)

    def softmax_joint(self, x_metric, y, prototypes, pos=True, neg=True):
        """
        - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))
        Note:
        log(Exp(y)) makes y always positive, which is required for our loss.
        """
        device = x_metric.device

        p_x, p_y = prototypes

        if torch.isnan(x_metric).any():
            print("skipping NaN batch")
            return torch.tensor(0)
        assert pos or neg, "At least one of the pos/neg terms must be activated in the Loss!"
        assert len(x_metric.shape) == 2, "Should only have batch and metric dimension."
        bs = x_metric.size(0)

        # All prototypes
        loss = None
        y_unique = torch.unique(y).squeeze()
        neg = False if len(y_unique.size()) == 0 else neg  # If only from the same class, there is no neg term
        y_unique = y_unique.view(-1)

        for label_idx in range(y_unique.size(0)):  # [summation over i]
            c = y_unique[label_idx]

            # Select from batch
            xc_idxs = (y == c).nonzero().squeeze(dim=1)
            xc = x_metric.index_select(0, xc_idxs)

            xk_idxs = (y != c).nonzero().squeeze(dim=1)
            xk = x_metric.index_select(0, xk_idxs)

            p_idx = (p_y == c).nonzero().squeeze(dim=1)
            pc = p_x[p_idx].detach()
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]]).detach()  # Other class prototypes

            lnL_pos = self.attractor(pc, pk, xc, device, include_batch=True) if pos else 0  # Pos
            lnL_neg = self.repellor(pc, pk, xc, xk, device, include_batch=True) if neg else 0  # Neg

            # Pos + Neg
            Loss_c = -lnL_pos - lnL_neg  # - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))

            # Update loss
            loss = Loss_c if loss is None else loss + Loss_c

            # Checks
            try:
                assert lnL_pos <= 0
                assert lnL_neg <= 0
                assert loss >= 0 and loss < 1e10
            except:
                exit(1)

        return loss / bs  # Make independent batch size

    def repellor(self, pc, pk, xc, xk, gpu, include_batch=True):
        # Gather per other-class samples
        if not include_batch:
            union_c = pc
        else:
            union_c = torch.cat([xc, pc])
        union_ck = torch.cat([union_c, pk]) #.clone().detach()
        c_split = union_c.shape[0]

        neg_Lterms = torch.mm(union_ck, xk.t()).div_(self.T).exp_()  # Last row is with own prototype
        pk_terms = neg_Lterms[c_split:].sum(dim=0).unsqueeze(0)  # For normalization
        pc_terms = neg_Lterms[:c_split]
        Pneg = pc_terms / (pc_terms + pk_terms + 1e-12)

        expPneg = (Pneg[:-1] + Pneg[-1].unsqueeze(0)) / 2  # Expectation pseudo/prototype
        lnPneg_k = expPneg.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        lnPneg = lnPneg_k.sum()  # Sum over (pseudo-prototypes), and instances

        try:
            assert -10e10 < lnPneg <= 0
        except:
            print("error")
            exit(1)
        return lnPneg

    def attractor(self, pc, pk, xc, gpu, include_batch=True):
        # Union: Current class batch-instances, prototype, memory
        if include_batch:
            pos_union_l = [xc.clone()]
            pos_len = xc.shape[0]
        else:
            pos_union_l = []
            pos_len = 1
        pos_union_l.append(pc)

        pos_union = torch.cat(pos_union_l)
        all_pos_union = torch.cat([pos_union, pk]).clone().detach()  # Include all other-class prototypes p_k
        pk_offset = pos_union.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms (don't include self => diagonal)
        pos_Lterms = torch.mm(all_pos_union, xc.t()).div_(self.T).exp_()  # .fill_diagonal_(0)
        if include_batch:
            mask = torch.eye(*pos_Lterms.shape).bool().to(pc.device)
            pos_Lterms = pos_Lterms.masked_fill(mask, 0)  # Fill with zeros

        Lc_pos = pos_Lterms[:pk_offset]
        Lk_pos = pos_Lterms[pk_offset:].sum(dim=0)  # sum column dist to pk's

        # Divide each of the terms by itself+ Lk term to get probability
        Pc_pos = Lc_pos / (Lc_pos + Lk_pos + 1e-12)
        expPc_pos = Pc_pos.sum(0) / (pos_len)  # Don't count self in
        lnL_pos = expPc_pos.log_().sum()

        # Sum instance loss-terms (per-column), divide by pk distances as well
        try:
            assert lnL_pos <= 0
        except:
            exit(1)
        return lnL_pos
