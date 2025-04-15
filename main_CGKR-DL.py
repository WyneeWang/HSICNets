from torch import optim
from Model import Network, OURS_Distiller
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Teacher model
class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = (self.gamma * out + x).unsqueeze(-1)
        return out
class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width, channle = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = self.gamma * out + x
        return out
class HsiModel(nn.Module):
    def __init__(self, band, p):
        super(HsiModel, self).__init__()
        self.p = p

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 5), stride=(1, 1, 2))
        self.batch_norm11 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv122 = nn.Conv3d(in_channels=6, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm122 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv123 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm123 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv124 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        kernel_3d = math.floor((band - 4) / 2)
        self.conv13 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.attention_spectral = CAM_Module(24)
        self.attention_spatial = PAM_Module(24)

        self.batch_norm_spectral = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish(),
                                                 nn.Dropout(p=0.5))
        self.batch_norm_spatial = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish(),
                                                nn.Dropout(p=0.5))

        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        self.attention = nn.Sequential(
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )  # MLP

        self.position = PositionModel()

    def forward(self, X, xp):

        X = X.unsqueeze(1)

        X11 = self.conv11(X)
        X11 = self.batch_norm11(X11)

        XS1 = torch.chunk(X11, 4, dim=1)  # 在给定维度(轴)上将输入张量进行分块儿
        X121 = XS1[0]
        X122 = self.conv122(XS1[1])
        X122 = self.batch_norm122(X122)
        X123 = torch.cat((X122, XS1[2]), dim=1)
        X123 = self.conv123(X123)
        X123 = self.batch_norm123(X123)
        X124 = torch.cat((X123, XS1[3]), dim=1)
        X124 = self.conv124(X124)
        X12 = torch.cat((X121, X122, X123, X124), dim=1)
        X12 = self.batch_norm12(X12)
        X13 = self.conv13(X12)

        X1 = self.attention_spectral(X13)
        X1 = torch.mul(X1, X13)

        X21 = self.conv21(X)
        X21 = self.batch_norm21(X21)

        XS2 = torch.chunk(X21, 4, dim=1)
        X221 = XS2[0]
        X222 = self.conv222(XS2[1])
        X222 = self.batch_norm222(X222)

        X223 = torch.cat((X222, XS2[2]), dim=1)
        X223 = self.conv223(X223)
        X223 = self.batch_norm223(X223)

        X224 = torch.cat((X223, XS2[3]), dim=1)
        X224 = self.conv224(X224)
        X22 = torch.cat((X221, X222, X223, X224), dim=1)
        X22 = self.batch_norm22(X22)

        X2 = self.attention_spatial(X22)
        X2 = torch.mul(X2, X22)

        X1 = self.batch_norm_spectral(X1)
        X1 = self.global_pooling(X1)
        X1 = X1.squeeze(-1).squeeze(-1).squeeze(-1)
        X2 = self.batch_norm_spatial(X2)
        X2 = self.global_pooling(X2)
        X2 = X2.squeeze(-1).squeeze(-1).squeeze(-1)

        #feature_X = torch.cat((X1, X2), dim=1)
        feature_X = X1 + X2
        attention = self.attention(feature_X)
        feature_X = attention * feature_X

        Xp = xp[:, :, :, :-2]
        Xl = xp[:, self.p, self.p, -2:]
        feature_Xp = self.position(Xp)
        feature = torch.cat((feature_X, feature_Xp), dim=1)
        return feature_X, Xl, feature
class LidarModel(nn.Module):
    def __init__(self, p):
        super(LidarModel, self).__init__()
        self.p = p

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.attention_spatial = PAM_Module(24)

        self.batch_norm_spatial = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish(),
                                                nn.Dropout(p=0.5))

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.position = PositionModel()

    def forward(self, X, xp):
        X = X.unsqueeze(1)

        X21 = self.conv21(X)
        X21 = self.batch_norm21(X21)

        XS2 = torch.chunk(X21, 4, dim=1)
        X221 = XS2[0]
        X222 = self.conv222(XS2[1])
        X222 = self.batch_norm222(X222)

        X223 = torch.cat((X222, XS2[2]), dim=1)
        X223 = self.conv223(X223)
        X223 = self.batch_norm223(X223)

        X224 = torch.cat((X223, XS2[3]), dim=1)
        X224 = self.conv224(X224)
        X22 = torch.cat((X221, X222, X223, X224), dim=1)
        X22 = self.batch_norm22(X22)

        X2 = self.attention_spatial(X22)
        X2 = torch.mul(X2, X22)

        X2 = self.batch_norm_spatial(X2)
        X2 = self.global_pooling(X2)
        X2 = X2.squeeze(-1).squeeze(-1).squeeze(-1)

        feature_X = X2

        Xp = xp[:, :, :, :-2]
        Xl = xp[:, self.p, self.p, -2:]
        feature_Xp = self.position(Xp)
        feature = torch.cat((feature_X, feature_Xp), dim=1)
        return feature_X, Xl, feature
class PositionModel(nn.Module):
    def __init__(self,):
        super(PositionModel, self).__init__()

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 2), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.attention_spatial = PAM_Module(24)

        self.batch_norm_spatial = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish(),
                                                nn.Dropout(p=0.5))

        self.global_pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, X):
        X = X.unsqueeze(1)

        X21 = self.conv21(X)
        X21 = self.batch_norm21(X21)

        XS2 = torch.chunk(X21, 4, dim=1)
        X221 = XS2[0]
        X222 = self.conv222(XS2[1])
        X222 = self.batch_norm222(X222)

        X223 = torch.cat((X222, XS2[2]), dim=1)
        X223 = self.conv223(X223)
        X223 = self.batch_norm223(X223)

        X224 = torch.cat((X223, XS2[3]), dim=1)
        X224 = self.conv224(X224)
        X22 = torch.cat((X221, X222, X223, X224), dim=1)
        X22 = self.batch_norm22(X22)

        X2 = self.attention_spatial(X22)
        X2 = torch.mul(X2, X22)

        X2 = self.batch_norm_spatial(X2)
        X2 = self.global_pooling(X2)
        X2 = X2.squeeze(-1).squeeze(-1).squeeze(-1)

        feature = X2
        return feature
# ---------------------------------------------------------------
class OURS_GCN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(OURS_GCN, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.f_psi = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid()
        )  # MLP
        # init
        self.init_weights_()

    def init_weights_(self,):
        self.weight.data.normal_(mean=0, std=0.01)
        self.bias.data.normal_(mean=0, std=0.01)

    def calculate(self, XH, XP, K, sigma=0.1):
        eps = np.finfo(float).eps
        sigma = torch.tensor(sigma, requires_grad=False).cuda()
        XH = XH / (sigma + eps)  # N*d
        WHo = torch.norm(XH[:, None] - XH, dim=2, p=2)
        # Min-Max scaling
        min_a = torch.min(WHo)
        max_a = torch.max(WHo)
        WH = torch.exp(-((WHo - min_a) / (max_a - min_a)) / 2)

        Wlo = torch.norm(XP[:, None] - XP, dim=2, p=2)
        # Min-Max scaling
        min_a = torch.min(Wlo)
        max_a = torch.max(Wlo)
        Wl = torch.exp(-((Wlo - min_a) / (max_a - min_a)))

        WW = WH + 0.2 * Wl

        topk, indices = torch.topk(WW, K)
        # keep top-k values
        mask = torch.zeros_like(WW)
        mask = mask.scatter(1, indices, 1)
        mask = mask + torch.t(mask)

        Adj = WW.clone()
        mask_p1 = torch.nonzero(mask, as_tuple=False)
        Adj[mask_p1[:, 0], mask_p1[:, 1]] = WW[mask_p1[:, 0], mask_p1[:, 1]] / WW[mask_p1[:, 0], mask_p1[:, 1]]
        mask_p0 = (mask == 0).nonzero()
        Adj[mask_p0[:, 0], mask_p0[:, 1]] = WW[mask_p0[:, 0], mask_p0[:, 1]] - WW[mask_p0[:, 0], mask_p0[:, 1]]

        return Adj

    def forward(self, inputs):
        XH, XP, XX, K = inputs
        W = self.calculate(XH, XP, K)  # 邻接矩阵
        x = torch.mm(XX, self.weight)  # 节点特征
        vi = x.unsqueeze(0).repeat(x.shape[0], 1, 1) * W.unsqueeze(2).repeat(1, 1, x.shape[1])  # (32,32,128)  #every起点，所对应的所有终点，包括自身
        #求关系矩阵
        vij = (x.unsqueeze(0).repeat(x.shape[0], 1, 1) + vi).reshape(-1, x.shape[1])  #(1024,128)
        att = (self.f_psi(vij)).reshape(x.shape[0], x.shape[0], -1)  #(32,32,128)
        self_vi = torch.zeros_like(att)
        for i in range(self_vi.shape[0]):
            self_vi[i, i, :] = torch.ones(x.shape[1])
        att = torch.where(self_vi > 0, self_vi, att)

        out = (att * vi).sum(axis=0) + self.bias

        return out[:, :XH.shape[1]], XP, out, K
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(GCN, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(OURS_GCN(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        XH, XP, XX, K = inputs
        out = self.layers((XH, XP, XX, K))
        return out
# ---------------------------------------------------------------
class Teacher(nn.Module):
    def __init__(self, INPUT_DIMENSION, CLASSES_NUM, PATCH, K):
        super(Teacher, self).__init__()
        self.p = int((PATCH - 1) / 2)
        self.K = K
        self.CHFeature = HsiModel(INPUT_DIMENSION, self.p)
        self.h_dim = 48
        self.l_dim = 48
        self.f_dim = 48
        self.GHFeature = GCN(self.h_dim, self.h_dim)
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.CLFeature = LidarModel(self.p)
        self.GLFeature = GCN(self.l_dim, self.l_dim)
        self.bn2 = nn.BatchNorm1d(self.l_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.f_dim, 1),
            nn.Sigmoid()
        )  # MLP
        self.fcH = nn.Linear(self.h_dim, CLASSES_NUM)
        self.fcL = nn.Linear(self.l_dim, CLASSES_NUM)
        self.fcF = nn.Linear(self.f_dim, CLASSES_NUM)

    def forward(self, xh, xl, xp):
        # CNN HSI
        XH, XP, XX = self.CHFeature(xh, xp)  # (32,258)
        # GCN HSI
        _, _, feature_GH, _ = self.GHFeature((XH, XP, XX, self.K))
        feature_GH = F.relu(self.bn1(feature_GH), inplace=True)
        # CNN LIDAR
        XL, XP, XX = self.CLFeature(xl, xp)  # (32,258)
        # GCN LIDAR
        _, _, feature_GL, _ = self.GLFeature((XL, XP, XX, self.K))
        feature_GL = F.relu(self.bn2(feature_GL), inplace=True)
        #feature_GF = torch.cat((feature_GH, feature_GL), 1)
        #  fusion
        feature_GF = feature_GH + feature_GL
        attention = self.attention(feature_GF)
        feature_GF = attention * feature_GF
        #  prediction
        outH = self.fcH(feature_GH)
        outL = self.fcL(feature_GL)
        outF = self.fcF(feature_GF)
        return feature_GF, outF
# ===============================================================
# Student HSI model
class StudentHsi(nn.Module):
    def __init__(self, INPUT_DIMENSION, CLASSES_NUM, PATCH, K):
        super(StudentHsi, self).__init__()
        self.p = int((PATCH - 1) / 2)
        self.K = K
        self.h_dim = 48
        self.CHFeature = HsiModel(INPUT_DIMENSION, self.p)
        self.GHFeature = GCN(self.h_dim, self.h_dim)
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.fcH = nn.Linear(self.h_dim, CLASSES_NUM)

    def forward(self, xh, xp):
        # CNN HSI
        XH, XP, XX = self.CHFeature(xh, xp)  # (32,258)
        # GCN HSI
        _, _, feature_GH, _ = self.GHFeature((XH, XP, XX, self.K))
        feature_GH = F.relu(self.bn1(feature_GH), inplace=True)
        #  prediction
        outH = self.fcH(feature_GH)
        return feature_GH, outH
# ===============================================================
# Distill model
class logits_D(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_hidden, self.n_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_class+1, bias=False)

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist
class local_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb):
        emb = F.normalize(emb, p=2)
        u = emb
        v = emb @ torch.diag(self.d)
        ## Compute a new edge feature named 'z' by a dot-product between the
        # source node feature 'ew' and destination node feature 'e'.
        pair_dis = torch.matmul(u, torch.transpose(v, dim0=0, dim1=1)).view(-1,)
        return pair_dis * self.scale
class global_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, summary):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        assert summary.shape[-1] == 1
        sim = sim @ summary
        return sim * self.scale
# ===============================================================

def distillation_loss(teacher_pred, student_pred, temperature=1.0):
    d_loss = F.kl_div(
        F.softmax(teacher_pred / temperature, dim=-1),
        F.softmax(student_pred / temperature, dim=-1)
    )
    return d_loss

def train_one_epoch(x1, x2, xp, y, teacher, student, DE, DG):

    student_optimizer = optim.Adam(student.parameters(), lr=0.0001)
    optimiser_D = torch.optim.Adam([{"params": Discriminator_e.parameters()}, {"params": Discriminator_g.parameters()}], lr=0.0001, weight_decay=5e-4)

    #  损失函数
    loss = torch.nn.CrossEntropyLoss()
    loss_dis = torch.nn.BCELoss()
    teacher.eval()
    student.train()

    x1, x2, xp, y = x1.cuda(), x2.cuda(), xp.cuda(), y.cuda()
    tea_emb, teacher_pred = teacher(x1, x2, xp)
    stu_emb, student_pred = student(x1, xp)

    # 损失
    student_loss = loss(student_pred, y.long())
    d_loss = distillation_loss(teacher_pred, student_pred)
    classes = torch.argmax(student_pred, dim=-1)
    student_accuracy = (classes == y).sum() / len(y)

    # ============================================
    #  Train Dis
    # ============================================
    # distinguish by De
    DE.train()
    pos_e = DE(tea_emb.detach())
    neg_e = DE(stu_emb.detach())
    real_e = torch.sigmoid(pos_e)
    fake_e = torch.sigmoid(neg_e)
    ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))  # 公式（1-1）

    # distinguish by Dg
    DG.train()
    tea_sum = torch.sigmoid(tea_emb.detach().mean(dim=0)).unsqueeze(-1)
    pos_g = DG(tea_emb.detach(), tea_sum)
    neg_g = DG(stu_emb.detach(), tea_sum)
    real_g = torch.sigmoid(pos_g)
    fake_g = torch.sigmoid(neg_g)
    ad_gloss1 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))  # 公式（1-2）

    stu_sum = torch.sigmoid(stu_emb.detach().mean(dim=0)).unsqueeze(-1)
    neg_g = DG(tea_emb.detach(), stu_sum)
    pos_g = DG(stu_emb.detach(), stu_sum)
    real_g = torch.sigmoid(pos_g)
    fake_g = torch.sigmoid(neg_g)
    ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))  # 公式（1-3）

    loss_D = ad_eloss + ad_gloss1 + ad_gloss2

    optimiser_D.zero_grad()
    loss_D.backward()
    optimiser_D.step()
    # ============================================
    #  Train Stu
    # ============================================
    ## to fool DE
    DE.eval()
    neg_e = DE(stu_emb)
    fake_e = torch.sigmoid(neg_e)
    ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))

    ## to fool DG
    DG.eval()
    tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
    neg_g = DG(stu_emb, tea_sum)
    fake_g = torch.sigmoid(neg_g)
    ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

    stu_sum = torch.sigmoid(stu_emb.mean(dim=0)).unsqueeze(-1)
    neg_g = DG(tea_emb, stu_sum)
    pos_g = DG(stu_emb, stu_sum)
    real_g = torch.sigmoid(pos_g)
    fake_g = torch.sigmoid(neg_g)
    ad_gloss2 = loss_dis(real_g, torch.zeros_like(real_g)) + loss_dis(fake_g, torch.ones_like(fake_g))

    loss_DG = ad_eloss + ad_gloss1 + ad_gloss2

    combined_loss = 0.6 * student_loss + 1.0 * d_loss + 1.0 * loss_DG

    student_optimizer.zero_grad()
    combined_loss.backward()
    student_optimizer.step()

    return {
        "student_loss": student_loss.cpu().detach().numpy(),
        "student_accuracy": student_accuracy.cpu().detach().numpy(),
        "distillation_loss": d_loss.cpu().detach().numpy(),
        "discriminator_loss": loss_DG.cpu().detach().numpy(),
        "combined_loss": combined_loss.cpu().detach().numpy(),
    }


if __name__ == "__main__":
    # train data
    xh = torch.randn(128, 5, 5, 64)  # BATCH_SIZE=128
    xl = torch.randn(128, 5, 5, 1)
    xp = torch.randn(128, 5, 5, 4)
    y = torch.randint(0, 11, (128,))  # CLASSES_NUM=11
    # model
    # teacher model
    teacher = Teacher(INPUT_DIMENSION=64, CLASSES_NUM=11, PATCH=5, K=60)
    teacher.cuda()
    teacher.load_state_dict(torch.load('checkpoints/best_model_teacher.pkl'))  #  Pretrained teacher model
    # student model
    student = StudentHsi(INPUT_DIMENSION=64, CLASSES_NUM=11, PATCH=5, K=60)
    student.cuda()
    # distiller model
    Discriminator_e = local_emb_D(n_hidden=48).cuda()
    Discriminator_g = global_emb_D(n_hidden=48).cuda()
    train_one_epoch(xh, xl, xp, y, teacher, student, Discriminator_e, Discriminator_g)











