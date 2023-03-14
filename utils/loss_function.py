import torch
import torch.nn as nn


class Multi_weighted_crossentropyloss(nn.Module):
    def __init__(self,
                 labels,
                 weighted=False,
                 label_weight=None,
                 device=None) -> None:
        super().__init__()

        self.labels = labels
        self.weighted = weighted
        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'

        self.label_weight = label_weight
        if self.label_weight is not None:
            assert len(labels) == len(label_weight)

        self.loss_list = nn.ModuleList()

        self._load_weight_dic()

        for label in labels:
            self.loss_list.append(self._get_loss(label))

    def _get_loss(self, label):
        if self.weighted:
            weight = torch.tensor(self.weight_dic[label],
                                  dtype=torch.float).to(self.device)
            weight = sum(weight) / weight
            weight = weight / sum(weight)
            return nn.CrossEntropyLoss(weight=weight)
        else:
            return nn.CrossEntropyLoss()

    def _load_weight_dic(self):
        self.weight_dic = {
            'tem': [1, 1.05],
            'san': [1, 1.63],
            'tem_or_san': [1, 1.7]
        }

    def forward(self, outputs, y):
        losses = []
        for out, lab, loss_func in zip(outputs, y, self.loss_list):
            losses.append(loss_func(out, lab))

        if self.label_weight is not None:
            losses_weight = []
            for loss, w_label in zip(losses, self.label_weight):
                losses_weight.append(loss * w_label)
            loss_sum = sum(losses_weight)
        else:
            loss_sum = sum(losses)

        return loss_sum