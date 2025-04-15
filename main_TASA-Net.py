from __future__ import print_function
import collections
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import torch.optim as optim

# visual model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, patch_size, n_bands, embed_dim):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2), padding=(0,1,1))
        self.conv1 = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=(1,3,3), bias=False)
        self.patch_size = patch_size
        self.fc = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.n_bands, self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0], -1)
            s = x.size()[1]
        return s

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.conv1(x)
        x = x.view(x.shape[0],-1)
        proj = self.fc(x)
        return proj
# ----------------------------------------------------------------------------------------------
# text model
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
# ----------------------------------------------------------------------------------------------
# VS-Prompts
class InjectionBlock(nn.Module):
    def __init__(self, embed_dim, n_cvx):
        super(InjectionBlock, self).__init__()
        self.n_cvx = n_cvx
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 16, embed_dim, bias=False),
            nn.Sigmoid()
        )

        self.linearlayer1 = nn.Sequential(nn.Linear((embed_dim * 2), embed_dim))
        self.linearlayer2 = nn.Sequential(nn.Linear((embed_dim * 2), embed_dim))
        self.gap = nn.AdaptiveAvgPool2d((1, 512))

        self.meta_net = nn.Sequential(*[
            nn.Sequential(collections.OrderedDict([
                ("linear1", nn.Linear(embed_dim, embed_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(embed_dim // 16, embed_dim))
            ])) for _ in range(self.n_cvx)
        ])

    def forward(self, vis):
        vis_f = self.gap(vis.unsqueeze(1))
        attn1 = self.attention(vis_f.type(torch.float))
        mulattn1 = torch.mul(attn1, vis_f)
        resattn1 = torch.cat((mulattn1, vis_f), 2)
        linear1 = self.linearlayer1(resattn1)

        attn2 = self.attention(linear1.type(torch.float))
        mulattn2 = torch.mul(attn2, vis_f)
        resattn2 = torch.cat((mulattn2, vis_f), 2)
        linear2 = self.linearlayer2(resattn2)

        output_vis = linear2.to(torch.float16)

        output = torch.cat([self.meta_net[i](output_vis.type(torch.float)) for i in range(self.n_cvx)], dim=1)
        return output
# ----------------------------------------------------------------------------------------------

# OUES MODEL
class OURSNet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 inchannel,
                 vision_patch_size: int,
                 num_classes,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 classnames: str,
                 num_visual_prompts
                 ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.classnames = classnames
        self.num_classes = num_classes

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.visual = D_Res_3d_CNN(1, 8, 16, vision_patch_size, inchannel, self.embed_dim)
        # ------------------------------------------------------------------------------------------------------------

        # TEXT Prompts 'A photo of a 【CLASS】'
        self.n_cvx = num_visual_prompts
        self.tokenized_prompts_ct, self.n_ctx, self.tokenized_prompts_cv = self.prompt_construct()
        # ------------------------------------------------------------------------------------------------------------
        # VISUAL PROMPTS
        # VG-VSP
        self.injection = InjectionBlock(embed_dim, self.n_cvx)

        # 动态调整融合系数γ
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # ------------------------------------------------------------------------------------------------------------

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_context(self, image_features):
        return self.injection(image_features)

    def prompt_construct(self):
        # TEXT PROMPTS
        # use given words to initialize context vectors
        ctx_init = 'A hyperspectral image of'
        # ctx_init = 'A photo of a'
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = torch.nonzero(clip.tokenize(ctx_init)).size(0) - 2
        classnames = [name.replace("_", " ") for name in self.classnames]
        prompts_ct = [ctx_init + " " + name + "." for name in classnames]
        # 单词向量token化 'A photo of a 【CLASS】'
        tokenized_prompts_ct = torch.cat([clip.tokenize(p) for p in prompts_ct])  # (n_cls, n_tkn)

        # VISUAL PROMPTS
        prompt_prefix_cv = " ".join(["X"] * self.n_cvx)
        classnames = [name.replace("_", " ") for name in self.classnames]
        prompts_cv = [prompt_prefix_cv + " " + name + "." for name in classnames]
        # 单词向量token化 'X X X X X X【CLASS】'
        tokenized_prompts_cv = torch.cat([clip.tokenize(p) for p in prompts_cv])  # (n_cls, n_tkn)

        return tokenized_prompts_ct, n_ctx, tokenized_prompts_cv

    def forward(self, image, label):

        logit_scale = self.logit_scale.exp()

        # Stage 0: 当前视觉图像的编码特征
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # -----------------------------------------------------------------------------------------------------------

        # Stage 1: 粗粒度文本text_features_ct
        prompts_ct = self.token_embedding(self.tokenized_prompts_ct.cuda())  # [n_cls, 77, d_model]
        text_features_ct = self.encode_text(prompts_ct, self.tokenized_prompts_ct)  # (n_cls, d_model)
        text_features_ct = text_features_ct.unsqueeze(0).expand(image_features.shape[0], -1, -1)

        # Stage 2: 细粒度文本text_features_cv
        context_features = self.encode_context(image_features)  # [batch, n_cvx, d_model]
        embedding = self.token_embedding(self.tokenized_prompts_cv.cuda())  # [n_cls, 77, d_model]
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_cvx:, :]
        ctx_shifted_expanded = context_features.unsqueeze(1).expand(-1, self.num_classes, -1, -1)
        prompts_cv = torch.cat([prefix.unsqueeze(0).expand(ctx_shifted_expanded.size(0), -1, -1, -1),
                                ctx_shifted_expanded,
                                suffix.unsqueeze(0).expand(ctx_shifted_expanded.size(0), -1, -1, -1)], dim=2)
        batch_size, n_cls, seq_len, embed_dim = prompts_cv.shape
        prompts_cv = prompts_cv.view(batch_size * n_cls, seq_len, embed_dim)
        tokenized_prompts = self.tokenized_prompts_cv.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size * n_cls, -1)
        text_features_cv = self.encode_text(prompts_cv, tokenized_prompts)  # (batch_size * n_cls, embed_dim)
        text_features_cv = text_features_cv.view(batch_size, n_cls, -1)

        # -----------------------------------------------------------------------------------------------------------

        # Stage 3: 融合粗细粒度文本语义text_features
        # 计算相似度 (使用矩阵乘法优化)
        text_features_ct_norm = text_features_ct / text_features_ct.norm(dim=-1, keepdim=True)
        text_features_cv_norm = text_features_cv / text_features_cv.norm(dim=-1, keepdim=True)
        similarity_ct = torch.einsum('bce,be->bc', text_features_ct_norm, image_features)
        similarity_cv = torch.einsum('bce,be->bc', text_features_cv_norm, image_features)

        # 计算增强权重 α_i 和抑制权重 β_i
        alpha = torch.exp(similarity_cv) / torch.sum(torch.exp(similarity_cv), dim=-1, keepdim=True)
        beta = torch.exp(similarity_ct) / torch.sum(torch.exp(similarity_ct), dim=-1, keepdim=True)

        # 计算动态融合系数 γ
        gamma = self.gamma_net(image_features).unsqueeze(-1)

        # 非线性加权融合特征
        text_features = gamma * (alpha.unsqueeze(-1) * text_features_cv) + (1 - gamma) * (beta.unsqueeze(-1) * text_features_ct)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # LOSS
        logits = logit_scale * torch.einsum('bce,be->bc', text_features, image_features)
        loss = F.cross_entropy(logits, label.long())

        return logits, loss
        # -----------------------------------------------------------------------------------------------------------


def train_one_epoch(data_src, label_src, model, episode=10000):
    LEARNING_RATE = 1e-3 / math.pow((1 + 10 * (episode - 1) / 200), 0.75)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.cuda()
    model.train()
    label_src_pred, loss = model(data_src.cuda(), label_src.cuda())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = label_src_pred.data.max(1)[1]
    CNN_correct = pred.eq(label_src.cuda().data.view_as(pred)).cpu().sum()

    return {
            "train_loss": loss.cpu().detach().numpy(),
            "train_accuracy": CNN_correct.cpu().detach().numpy(),
        }


if __name__ == "__main__":
    # train data
    data_src = torch.randn(7, 64, 9, 9)  # BATCH_SIZE=128
    label_src = torch.randint(0, 7, (7,))  # CLASSES_NUM=7
    label_values = ["grass healthy", "grass stressed", "trees",
                    "water", "residential buildings",
                    "non-residential buildings", "road"]

    # model
    pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict ["text_projection"].shape[1]
    context_length = pretrained_dict ["positional_embedding"].shape[0]
    vocab_size = pretrained_dict ["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict ["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3
    patch_size = 7
    num_visual_prompts = 4
    model = OURSNet(embed_dim, data_src.size(1), patch_size, len(label_values), context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, label_values, num_visual_prompts)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')

    # TRAINING
    train_one_epoch(data_src, label_src, model)