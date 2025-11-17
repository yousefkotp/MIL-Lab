from src.models.mil_template import MIL
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
from src.models.layers import create_mlp

MODEL_TYPE = 'meanmil'


class MeanMIL(MIL):
    """
    Mean MIL model: embeds instances with an optional MLP and aggregates by simple average.

    Args:
        in_dim (int): Input feature dimension per instance.
        embed_dim (int): Embedding dimension after the optional MLP. If num_fc_layers == 0, embed_dim is ignored.
        num_fc_layers (int): Number of FC layers in the instance embedding MLP. 0 disables the MLP.
        dropout (float): Dropout rate in the MLP.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__(in_dim=in_dim, embed_dim=(embed_dim if num_fc_layers > 0 else in_dim), num_classes=num_classes)
        self.use_mlp = num_fc_layers > 0
        if self.use_mlp:
            self.patch_embed = create_mlp(
                in_dim=in_dim,
                hid_dims=[embed_dim] * (num_fc_layers - 1),
                dropout=dropout,
                out_dim=embed_dim,
                end_with_fc=False,
            )
        else:
            self.patch_embed = nn.Identity()

        if num_classes > 0:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        # MeanMIL does not use attention; return zero-sized tensor to fit the interface.
        B, M, _ = MIL.ensure_batched(h).shape
        return torch.zeros((B, 1, M), device=h.device, dtype=h.dtype)

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, dict]:
        h = self.patch_embed(h)
        # simple masked mean if mask provided via log_dict hint (not used here)
        wsi_feats = h.mean(dim=1)  # [B, D]
        log_dict = {'attention': None if not return_attention else torch.zeros((h.shape[0], 1, h.shape[1]), device=h.device, dtype=h.dtype)}
        return wsi_feats, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(
        self,
        h: torch.Tensor,
        loss_fn: nn.Module = None,
        label: torch.LongTensor = None,
        attn_mask=None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ) -> tuple[dict, dict]:
        wsi_feats, log_dict = self.forward_features(h, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict


class MeanMILConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.0,
        in_dim: int = 1024,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.auto_map = {
            'AutoConfig': 'modeling_meanmil.MeanMILConfig',
            'AutoModel': 'modeling_meanmil.MeanMILModel',
        }


class MeanMILModel(PreTrainedModel):
    config_class = MeanMILConfig

    def __init__(self, config: MeanMILConfig, **kwargs):
        # allow overriding config fields via kwargs
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)
        super().__init__(config)
        self.model = MeanMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(MeanMILConfig.model_type, MeanMILConfig)
AutoModel.register(MeanMILConfig, MeanMILModel)

