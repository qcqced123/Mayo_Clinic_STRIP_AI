import torch, timm
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import freeze, reinit_topk


class STRIPModel(nn.Module):
    """ Baseline Model Class for Classification """
    def __init__(self, cfg):
        super(STRIPModel, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            features_only=False,
        )
        self.classifier = nn.Linear(self.model.config.hidden_size, cfg.num_classes)
