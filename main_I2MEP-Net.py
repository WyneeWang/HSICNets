import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_3Dblock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_3Dblock, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_3Dblock(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_3Dblock(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

    def forward(self, x):  # x:(400,100,9,9)
        x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)
        x = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x)  # (1,16,25,5,5)
        x = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.conv(x)  # (1,32,5,1,1)
        x = x.view(x.shape[0], -1)  # (1,160)
        return x
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x
class HsiNetwork(nn.Module):
    def __init__(self, SRC_INPUT_DIMENSION, TAR_INPUT_DIMENSION, N_DIMENSION, CLASS_NUM, FEATURE_DIM=160):
        super(HsiNetwork, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1, 8, 16)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.classifier = nn.Linear(in_features=FEATURE_DIM, out_features=CLASS_NUM)

    def forward(self, x, domain='source'):  # x
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        feature = self.feature_encoder(x)  # (45, 64)
        out = self.classifier(feature)
        return feature, out
def conv2x2(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_2Dblock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_2Dblock, self).__init__()

        self.conv1 = conv2x2(in_channel, out_channel)
        self.conv2 = conv2x2(out_channel, out_channel)
        self.conv3 = conv2x2(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out
class D_Res_2d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_2d_CNN, self).__init__()

        self.block1 = residual_2Dblock(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1), stride=(2, 2))
        self.block2 = residual_2Dblock(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.conv = nn.Conv2d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

    def forward(self, x):  # x:(400,1,9,9)
        # x = x.unsqueeze(1) # (400,1,9,9)
        x = self.block1(x)  # (1,8,9,9)
        x = self.maxpool1(x)  # (1,8,5,5)
        x = self.block2(x)  # (1,16,5,5)
        x = self.maxpool2(x)  # (1,16,3,3)
        x = self.conv(x)  # (1,32,1,1)
        x = x.view(x.shape[0], -1)  # (1,32)
        return x
class LidarNetwork(nn.Module):
    def __init__(self, CLASS_NUM, FEATURE_DIM=160):
        super(LidarNetwork, self).__init__()
        self.feature_encoder = D_Res_2d_CNN(1, 8, 16)
        self.generator = GeneratorNetwork()
        self.classifier = nn.Linear(in_features=FEATURE_DIM, out_features=CLASS_NUM)

    def forward(self, x):  # x
        feature = self.feature_encoder(x)  # (45, 64)
        feature = self.generator(feature)
        out = self.classifier(feature)
        return feature, out
class GeneratorNetwork(nn.Module):
    def __init__(self, FEATURE_DIM=160):
        super(GeneratorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=FEATURE_DIM, out_channels=FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(FEATURE_DIM),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):  # x
        noise = torch.randn(x.shape[0], 128).cuda()
        x = torch.cat((x, noise), 1)  # ,160
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        return x
# -------------------------------------------------------------------------------------------------------------------- #
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits
# -------------------------------------------------------------------------------------------------------------------- #
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
#域判别器
class DiscriminatorNetwork(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DiscriminatorNetwork, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1)  # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10, 10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
# -------------------------------------------------------------------------------------------------------------------- #
class Diversity_Net(nn.Module):
    def __init__(self, CLASS_NUM, FEATURE_DIM=160):
        super(Diversity_Net, self).__init__()
        self.num_classes = CLASS_NUM
        self.add_info = AddInfo()
        self.generator = Generator()
        self.fc = nn.Linear(FEATURE_DIM, self.num_classes, bias=False)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.ce = nn.CrossEntropyLoss().cuda()

    def forward(self, B1=None, B2=None, A=None, Y=None):
        add_info = self.add_info(A, B1, B2)
        A_rebuild = self.generator(add_info)
        A_rebuild = self.l2_norm(A_rebuild)
        logits = self.fc(A_rebuild * self.s)
        labels = torch.tensor(Y).expand(self.num_classes)
        score = self.ce(logits, torch.as_tensor(labels, dtype=torch.long).cuda())
        return A_rebuild, score

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output
class AddInfo(nn.Module):
    def __init__(self, FEATURE_DIM=160):
        super(AddInfo, self).__init__()
        self.dense = nn.Linear(FEATURE_DIM, 1024)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.f_psi = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.Sigmoid()
        )  # MLP

    def forward(self, A, B1, B2):
        A = self.dense(A)
        A = self.activation(A)
        B1 = self.dense(B1)
        B1 = self.activation(B1)
        B2 = self.dense(B2)
        B2 = self.activation(B2)
        # 计算A与B1、B2之间的差异
        diff_B1 = torch.abs(A - B1)
        diff_B2 = torch.abs(A - B2)
        # 找到差异最大的B1或B2
        max_diff = torch.max(torch.max(diff_B1), torch.max(diff_B2))
        if torch.max(diff_B1) == max_diff:
            sim = (B1.unsqueeze(2) @ A.reshape(1, -1)).mean(2)  # 属性相似度度量
        else:
            sim = (B2.unsqueeze(2) @ A.reshape(1, -1)).mean(2)  # 属性相似度度量
        att = self.f_psi(sim)
        out = A + att * (B1 - B2)  # 加权重
        out = self.dropout(out)
        return out
class Generator(nn.Module):
    def __init__(self, FEATURE_DIM=160):
        super(Generator, self).__init__()
        self.dense = nn.Linear(1024, FEATURE_DIM)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.dense(x)
        out = self.activation(out)
        out = self.dropout(out)
        return out
# -------------------------------------------------------------------------------------------------------------------- #

def train_one_epoch(supports, support_labels, querys, query_labels, auxiliary_data, auxiliary_labels, diversity_data, diversity_labels,
                    feature_encoder1, feature_encoder2, random_layer, discriminator, diversityNet,
                    CLASS_NUM, SHOT_NUM_PER_CLASS=1, FEATURE_DIM=160, episode=1):

    feature_encoder1.cuda()
    feature_encoder1.train()
    feature_encoder2.cuda()
    feature_encoder2.train()
    random_layer.cuda()
    discriminator.cuda()
    discriminator.train()
    diversityNet.cuda()
    diversityNet.train()
    # optimizer
    feature_encoder1_optim = torch.optim.Adam(feature_encoder1.parameters(), lr=0.001)
    feature_encoder2_optim = torch.optim.Adam(feature_encoder2.parameters(), lr=0.001)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    diversityNet_optim = torch.optim.Adam(diversityNet.parameters(), lr=0.001)
    #
    crossEntropy = nn.CrossEntropyLoss().cuda()
    criterion = torch.nn.MSELoss().cuda()

    total_hit, total_num = 0.0, 0.0

    '''Few-shot claification'''
    # calculate features
    support_features, support_logits = feature_encoder1(supports.cuda(), domain='source')
    query_features, query_logits = feature_encoder1(querys.cuda(), domain='source')
    auxiliary_features, auxiliary_logits = feature_encoder2(auxiliary_data.cuda())
    div_features, div_logits = feature_encoder1(diversity_data.cuda(), domain='target')
    # ---------------------------------------------------------------------------- #
    #  模态间跨模态多样性增强
    zg_support_features = (support_features + 2 * auxiliary_features) / (1 + 2)
    #
    #  模态内多样性增强
    #  利用div_features增强zg_support_features
    zg = torch.split(div_features, 2)
    zg_1 = torch.tensor([item.cpu().detach().numpy() for item in [split[0, :] for split in zg]]).cuda()
    zg_2 = torch.tensor([item.cpu().detach().numpy() for item in [split[1, :] for split in zg]]).cuda()
    # feature
    weight = torch.zeros((CLASS_NUM, FEATURE_DIM), requires_grad=True).cuda()
    gen_loss = 0.0
    for i in range(CLASS_NUM):
        weight_point = torch.zeros(SHOT_NUM_PER_CLASS * (CLASS_NUM + 1), FEATURE_DIM)
        for j in range(SHOT_NUM_PER_CLASS):
            gen_feature, gen_logits = diversityNet(zg_1, zg_2, zg_support_features[i * SHOT_NUM_PER_CLASS + j],
                                                   support_labels[i * SHOT_NUM_PER_CLASS + j])
            gen_feature = gen_feature + zg_support_features[i * SHOT_NUM_PER_CLASS + j]  # 残差操作 +
            features = torch.cat((gen_feature, zg_support_features[i * SHOT_NUM_PER_CLASS + j].unsqueeze(0)))
            weight_point[j * (CLASS_NUM + 1):(j + 1) * (CLASS_NUM + 1)] = features
            gen_loss += gen_logits
        weight[i] = torch.mean(weight_point, 0)
    support_pro = weight
    gen_loss /= (SHOT_NUM_PER_CLASS * CLASS_NUM)
    #
    # ---------------------------------------------------------------------------- #
    #  FSL classification loss
    logits = euclidean_metric(query_features, support_pro)
    fsl_loss = crossEntropy(logits, torch.as_tensor(query_labels, dtype=torch.long).cuda())
    #  辅助集分类损失
    sup_loss = crossEntropy(support_logits, torch.as_tensor(support_labels, dtype=torch.long).cuda())
    que_loss = crossEntropy(query_logits, torch.as_tensor(query_labels, dtype=torch.long).cuda())
    aux_loss = crossEntropy(auxiliary_logits, torch.as_tensor(auxiliary_labels, dtype=torch.long).cuda())
    div_loss = crossEntropy(div_logits, torch.as_tensor(diversity_labels, dtype=torch.long).cuda())
    extra_loss = sup_loss + que_loss + aux_loss + div_loss + gen_loss
    # ---------------------------------------------------------------------------- #
    #  S=1, A、T=0
    features1 = torch.cat([support_features, query_features, auxiliary_features], dim=0)
    outputs1 = torch.cat((support_logits, query_logits, auxiliary_logits), dim=0)
    softmax_output1 = nn.Softmax(dim=1)(outputs1)
    # set label: source 1; auxiliary_features 0
    domain_label1 = torch.zeros(
        [support_features.shape[0] + query_features.shape[0] + auxiliary_features.shape[0], 1]).cuda()
    domain_label1[:support_features.shape[0] + query_features.shape[0]] = 1
    randomlayer_out1 = random_layer.forward([features1, softmax_output1])
    domain_logits1 = discriminator(randomlayer_out1, episode)
    domain_loss1 = criterion(domain_logits1, domain_label1)
    #
    features2 = torch.cat([support_features, query_features, div_features], dim=0)
    outputs2 = torch.cat((support_logits, query_logits, div_logits), dim=0)
    softmax_output2 = nn.Softmax(dim=1)(outputs2)
    # set label: source 1; target div_features 0
    domain_label2 = torch.zeros([support_features.shape[0] + query_features.shape[0] + div_features.shape[0], 1]).cuda()
    domain_label2[:support_features.shape[0] + query_features.shape[0]] = 1
    randomlayer_out2 = random_layer.forward([features2, softmax_output2])
    domain_logits2 = discriminator(randomlayer_out2, episode)
    domain_loss2 = criterion(domain_logits2, domain_label2)
    domain_loss = domain_loss1 + domain_loss2

    '''Total loss'''
    loss = fsl_loss + 0.2 * extra_loss + domain_loss

    # Update parameters
    feature_encoder1.zero_grad()
    feature_encoder2.zero_grad()
    diversityNet.zero_grad()
    discriminator.zero_grad()
    loss.backward()
    feature_encoder1_optim.step()
    feature_encoder2_optim.step()
    diversityNet_optim.step()
    discriminator_optim.step()

    total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
    total_num += query_labels.shape[0]


if __name__ == "__main__":
    # source HS data
    supports = torch.randn(5, 144, 9, 9)  # [5 source classes ✖ 1 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    support_labels = torch.arange(5)
    querys = torch.randn(95, 144, 9, 9)  # [5 source classes ✖ 19 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    query_labels = torch.repeat_interleave(torch.arange(5), 19)
    # source LIDAR data as auxiliary data
    auxiliary_data = torch.randn(5, 1, 9, 9)  # [5 source classes ✖ 1 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    auxiliary_labels = torch.arange(5)
    # target few-shot HSI data as diversity data
    diversity_data = torch.randn(10, 103, 9, 9)  # [5 target classes ✖ 2 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    diversity_labels = torch.repeat_interleave(torch.arange(5), 2)

    # model
    feature_encoder1 = HsiNetwork(SRC_INPUT_DIMENSION=144, TAR_INPUT_DIMENSION=103, N_DIMENSION=100, CLASS_NUM=5)
    feature_encoder2 = LidarNetwork(CLASS_NUM=5)
    random_layer = RandomLayer([160, 5], 1024)
    discriminator = DiscriminatorNetwork()
    diversityNet = Diversity_Net(CLASS_NUM=5)

    # TRAINING
    train_one_epoch(supports, support_labels, querys, query_labels, auxiliary_data, auxiliary_labels, diversity_data, diversity_labels, feature_encoder1, feature_encoder2, random_layer, discriminator, diversityNet, CLASS_NUM=5)