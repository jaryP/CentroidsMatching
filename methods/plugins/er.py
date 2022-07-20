import torch
from avalanche.training.plugins import StrategyPlugin
from torch import cosine_similarity
from torch.utils.data import DataLoader


class EmbeddingRegularizationPlugin(StrategyPlugin):
    def __init__(self, mem_size: int, penalty_weight: float, distance: str = 'euclidean'):
        super().__init__()

        self.patterns_per_experience = mem_size
        self.penalty_weight = penalty_weight
        assert distance in ['cosine', 'euclidean']

        self.distance = distance

        self.memory_x, self.memory_y, self.memory_tid = {}, {}, {}

    def after_training_exp(self, strategy, **kwargs):
        self.update_memory(strategy,
                           strategy.experience.dataset,
                           strategy.clock.train_exp_counter,
                           strategy.train_mb_size)

        tid = strategy.experience.current_experience
        for param in strategy.model.classifier.classifiers[str(tid)].parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_memory(self, strategy, dataset, t, batch_size):
        dataloader = DataLoader(dataset.eval(),
                                batch_size=batch_size,
                                shuffle=True)
        tot = 0
        device = strategy.device
        strategy.model.eval()

        for mbatch in dataloader:
            x, y, tid = mbatch[0].to(device), mbatch[1], mbatch[-1].to(device)
            emb = strategy.model.feature_extractor(x, tid)

            emb = emb.clone().cpu()
            x = x.clone().cpu()
            tid = tid.clone().cpu()

            if tot + x.size(0) <= self.patterns_per_experience:
                if t not in self.memory_x:
                    self.memory_x[t] = x
                    self.memory_y[t] = emb
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], emb), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid),
                                                   dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff]
                    self.memory_y[t] = emb[:diff]
                    self.memory_tid[t] = tid[:diff]
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]),
                                                 dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], emb[:diff]),
                                                 dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t],
                                                    tid[:diff]), dim=0)
                break
            tot += x.size(0)

    def before_backward(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            total_dis = 0
            device = strategy.device

            for t in self.memory_x.keys():
                xref, past_emb, t = self.memory_x[t], self.memory_y[t], self.memory_tid[t]

                xref = xref.to(device)
                past_emb = past_emb.to(device)

                embedding = strategy.model.feature_extractor(xref, t)

                if self.distance == 'cosine':
                    sim = cosine_similarity(past_emb, embedding)
                    diff = 1 - sim
                else:
                    diff = torch.norm(past_emb - embedding, 2, -1)

                diff = diff.mean()
                total_dis += diff

            total_dis = total_dis / len(self.memory_x)

            strategy.loss += total_dis * self.penalty_weight
