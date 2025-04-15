import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# model
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
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
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
class Network(nn.Module):
    def __init__(self, SRC_INPUT_DIMENSION, TAR_INPUT_DIMENSION, N_DIMENSION, PATCH):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        # domain specific
        self.cix = torch.cat((self.create_domain_specific(1, PATCH), self.create_domain_specific(0, PATCH)), 0)  # (1,0)
        self.cit = torch.cat((self.create_domain_specific(0, PATCH), self.create_domain_specific(1, PATCH)), 0)  # (0,1)

    def create_domain_specific(self, fill_value, class_num):
        if fill_value == 1:
            return torch.ones(class_num, class_num).unsqueeze(0)
        else:
            return torch.zeros(class_num, class_num).unsqueeze(0)

    def forward(self, x, domain='source'):  # x
        # print(x.shape)
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
            cit_ = torch.tensor(np.repeat(self.cit.unsqueeze(0).numpy(), x.shape[0], axis=0)).type(torch.FloatTensor).cuda()  # (9,2,9,9)
            xc = torch.cat((x, cit_), 1)

        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
            cix_ = torch.tensor(np.repeat(self.cix.unsqueeze(0).numpy(), x.shape[0], axis=0)).type(torch.FloatTensor).cuda()  # (9,2,9,9)
            xc = torch.cat((x, cix_), 1)
        #xc:torch.Size([9, 102, 9, 9])
        feature = self.feature_encoder(xc)
        return feature
class DomainClassifier(nn.Module):
    def __init__(self, input_nc=160, output_nc=160):
        super(DomainClassifier, self).__init__()
        self.liner1 = nn.Linear(in_features=input_nc, out_features=input_nc // 2)
        self.liner2 = nn.Linear(in_features=input_nc // 2, out_features=output_nc)

    def forward(self, x):
        x_ci = F.relu(self.liner1(x), inplace=True)
        x_ci = F.relu(self.liner2(x_ci), inplace=True)
        return x_ci
class Metric_encoder(nn.Module):
    """docstring for RelationNetwork为每个类学习一个类注意力权重"""
    def __init__(self, CLASS_NUM):
        super(Metric_encoder, self).__init__()
        self.inchannel = 160 + CLASS_NUM
        self.f_psi = nn.Sequential(
            nn.Linear(self.inchannel, 160),
            nn.BatchNorm1d(160),
            nn.Sigmoid()
        )#MLP
    def forward(self, s, q, sl):#q(304,160) s(16,160) sl(16,)
        sl = torch.as_tensor(sl, dtype=torch.long).cuda()
        sl = F.one_hot(sl) #(16,16)
        ind = torch.cat((s, sl), 1) #(16,176)
        weight_ = self.f_psi(ind) #(16,160)
        attention_ = weight_.unsqueeze(0).repeat(q.shape[0], 1, 1)
        match_ = euclidean_metric(q, s)  # (304,16,160)
        attention_match_score = torch.mul(attention_, match_)  # (304,16,160)
        score = torch.sum(attention_match_score.contiguous(), dim=2)  # (304,16)
        return score
def euclidean_metric(a, b):

    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2)
    return logits


def train_one_epoch(supports1, support_labels1, querys1, query_labels1, supports2, support_labels2, querys2, query_labels2, feature_encoder, domain_classifier, metric_net_encoder):
    feature_encoder.cuda()
    domain_classifier.cuda()
    metric_net_encoder.cuda()
    feature_encoder.train()
    domain_classifier.train()
    metric_net_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=0.001)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=0.001)
    metric_net_encoder_optim = torch.optim.Adam(metric_net_encoder.parameters(), lr=0.001)

    crossEntropy = nn.CrossEntropyLoss().cuda()
    criterion = torch.nn.MSELoss().cuda()
    total_hit, total_num = 0.0, 0.0

    '''Few-shot claification'''
    # calculate features
    support_features1 = feature_encoder(supports1.cuda())
    query_features1 = feature_encoder(querys1.cuda())

    support_features2 = feature_encoder(supports2.cuda(), domain='target')
    query_features2 = feature_encoder(querys2.cuda(), domain='target')

    # STEP 1
    logits1 = metric_net_encoder(support_features1, query_features1, support_labels1)
    query_labels1 = torch.as_tensor(query_labels1, dtype=torch.long)
    CE_loss1 = crossEntropy(logits1, query_labels1.cuda())
    logits2 = metric_net_encoder(support_features2, query_features2, support_labels2)
    query_labels2 = torch.as_tensor(query_labels2, dtype=torch.long)
    CE_loss2 = crossEntropy(logits2, query_labels2.cuda())
    target_features = torch.cat((support_features2, query_features2), 0)
    reconstructed_t = domain_classifier(target_features)
    F_AET_loss = criterion(reconstructed_t, target_features)
    loss1 = CE_loss1 + CE_loss2 + 0.7 * F_AET_loss

    # Update parameters
    feature_encoder.zero_grad()
    metric_net_encoder.zero_grad()
    loss1.backward()
    feature_encoder_optim.step()
    metric_net_encoder_optim.step()

    # STEP 2
    support_features1 = feature_encoder(supports1.cuda())
    query_features1 = feature_encoder(querys1.cuda())
    source_features = torch.cat((support_features1, query_features1), 0)
    reconstructed_s = domain_classifier(source_features)
    AES_loss = criterion(reconstructed_s, source_features)
    support_features2 = feature_encoder(supports2.cuda(), domain='target')
    query_features2 = feature_encoder(querys2.cuda(), domain='target')
    target_features = torch.cat((support_features2, query_features2), 0)
    reconstructed_t = domain_classifier(target_features)
    AET_loss = criterion(reconstructed_t, target_features)
    if (0.5 - AET_loss) > 0:
        compare_loss = 0.5 - AET_loss
    else:
        compare_loss = 0
    loss2 = AES_loss + compare_loss

    # Update parameters
    domain_classifier.zero_grad()
    loss2.backward()
    domain_classifier_optim.step()

    logits = torch.cat((logits1, logits2),0)
    query_labels = torch.cat((query_labels1, query_labels2),0)
    total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
    total_num += query_labels.shape[0]


if __name__ == "__main__":
    # source data
    supports1 = torch.randn(5, 144, 9, 9)  # [5 source classes ✖ 1 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    support_labels1 = torch.arange(5)
    querys1 = torch.randn(95, 144, 9, 9)  # [5 source classes ✖ 19 per class, SRC_INPUT_DIMENSION, PATCH, PATCH]
    query_labels1 = torch.repeat_interleave(torch.arange(5), 19)
    # target few-shot data
    supports2 = torch.randn(5, 103, 9, 9)  # [5 target classes ✖ 1 per class, TAR_INPUT_DIMENSION, PATCH, PATCH]
    support_labels2 = torch.arange(5)
    querys2 = torch.randn(95, 103, 9, 9)  # [5 target classes ✖ 19 per class, TAR_INPUT_DIMENSION, PATCH, PATCH]
    query_labels2 = torch.repeat_interleave(torch.arange(5), 19)
    # model
    feature_encoder = Network(SRC_INPUT_DIMENSION=144, TAR_INPUT_DIMENSION=103, N_DIMENSION=100, PATCH=9)
    domain_classifier = DomainClassifier()
    metric_net_encoder = Metric_encoder(CLASS_NUM=5)
    # TRAINING
    train_one_epoch(supports1, support_labels1, querys1, query_labels1, supports2, support_labels2, querys2, query_labels2, feature_encoder, domain_classifier, metric_net_encoder)