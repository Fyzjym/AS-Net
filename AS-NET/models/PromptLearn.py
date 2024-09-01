import torch
import torch.nn as nn
import torch.nn.functional as F

"""class MetaNet(nn.Module):
    def __init__(self, vis_dim=512, ctx_dim=1):
        super().__init__()

        # self.linear1 = nn.Linear(vis_dim, vis_dim // 2)
        # self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(vis_dim // 2, ctx_dim)

        self.conv1d_1 = nn.Conv1d(vis_dim, vis_dim // 4, kernel_size=1, stride=1, padding=0)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(vis_dim // 4, vis_dim // 8, kernel_size=1, stride=1, padding=0)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv1d_3 = nn.Conv1d(vis_dim // 8, ctx_dim, kernel_size=1, stride=1, padding=0)

        # Kaiming
        nn.init.kaiming_normal_(self.conv1d_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1d_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1d_3.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        B, N, C = x.shape
        x = x.mean(1, keepdim=True) # B, 1, C
        print(x.shape)

        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.conv1d_2(x)
        x = self.relu_1(x)
        x = self.conv1d_3(x)
        x = x.view(B, -1, C)

        print(x.shape)
        return x
"""

# V1 QR shape not always same
"""

class PromptLearner(nn.Module):
    def __init__(self, n_ctx=4, res_flag=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.res_flag = res_flag

        embedding = torch.empty(8, 16, 512)
        nn.init.normal_(embedding, std=0.02)

        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        # print(ctx_vectors.shape)
        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = MetaNet(ctx_dim=self.n_ctx)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS


    def construct_prompts(self, ctx, prefix, suffix):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix
            ],
            dim=1
        )
        return prompts

    def forward(self, im_features, QR):

        B, N, C = im_features.shape
        bias = self.meta_net(im_features)  # 8, 4, 512
        # print(bias.shape)
        ctx = self.ctx  # (n_ctx, ctx_dim) 4, 512
        # print(ctx.shape)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(B, -1, -1) # torch.Size([8, 4, 512])

        # print(ctx.shape)

        prefix = self.token_prefix # 8 1 512
        suffix = self.token_suffix # 8 11 512
        # print(prefix.shape)
        # print(suffix.shape)


        ctx_bias = ctx + bias
        # print(ctx_bias.shape)

        prompts = self.construct_prompts(ctx_bias, prefix, suffix)  # same with QR
        # print(prompts.shape)

        if self.res_flag == True:
            prompts = prompts + QR

        return prompts


"""




## V2
"""

class MetaNet(nn.Module):
    def __init__(self, vis_dim=512, meta_dim=1):
        super(MetaNet, self).__init__()

        # Conv1d layers to reduce channel dimension
        self.conv1d_1 = nn.Conv1d(vis_dim, vis_dim // 4, kernel_size=1, stride=1, padding=0)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(vis_dim // 4, vis_dim // 8, kernel_size=1, stride=1, padding=0)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv1d_3 = nn.Conv1d(vis_dim // 8, meta_dim, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Kaiming initialization
        nn.init.kaiming_normal_(self.conv1d_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1d_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1d_3.weight, mode='fan_out', nonlinearity='relu')

        # Global Average Pooling layer to reduce length dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):


        x = x.permute(0, 2, 1)  # Change shape to [B, C, N] for Conv1d

        # Apply Global Average Pooling
        x = self.global_avg_pool(x)  # Now x is [B, 1, 1]

        x = self.conv1d_1(x)
        x = self.relu_1(x)

        x = self.conv1d_2(x)
        x = self.relu_2(x)

        x = self.conv1d_3(x)
        x = self.sigmoid(x)

        return x

class PromptLearner(nn.Module):
    def __init__(self, n_ctx=1, res_flag=False, mul_op = True):
        super().__init__()
        self.n_ctx = n_ctx
        self.res_flag = res_flag
        self.mul_op = mul_op

        self.meta_net = MetaNet()

    def construct_prompts(self, QR, st_bias):

        # prompts = torch.cat(
        #     [
        #         ctx,
        #         st_bias
        #     ],
        #     dim=1
        # )

        if self.mul_op == False:
            prompts = torch.add(QR, st_bias)


        if self.mul_op == True:
            prompts = QR * st_bias + QR


        return prompts

    def forward(self, im_features, QR):


        bias = self.meta_net(im_features)  # 8, 1, 1

        prompts = self.construct_prompts(QR, bias)  # same with QR

        # if self.res_flag == True:
        #     prompts = (prompts + QR) / 2

        assert QR.shape == prompts.shape

        return prompts

"""

class MetaNet(nn.Module):
    def __init__(self, vis_dim=90, meta_dim=1):
        super(MetaNet, self).__init__()

        # Conv1d layers to reduce channel dimension
        self.conv1d_1 = nn.Conv1d(vis_dim, vis_dim // 3, kernel_size=1, stride=1, padding=0)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(vis_dim // 3, meta_dim, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Kaiming initialization
        nn.init.kaiming_normal_(self.conv1d_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1d_2.weight, mode='fan_out', nonlinearity='relu')

        # Global Average Pooling layer to reduce length dimension
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):


        x = x.permute(1, 0, 2)  #


        # Apply Global Average Pooling
        # x = self.global_avg_pool(x)  # Now x is [B, 1, 1]
        # print(x.shape)

        x = self.conv1d_1(x)
        x = self.relu_1(x)


        x = self.conv1d_2(x)
        x = self.sigmoid(x)


        return x # 8 1 512

class PromptLearner(nn.Module):
    def __init__(self, n_ctx=1, res_flag=False, mul_op = 1):
        super().__init__()
        self.n_ctx = n_ctx
        self.res_flag = res_flag
        self.mul_op = mul_op # 0 = none, 1,2,3,4,5
        self.meta_net = MetaNet()

    def construct_prompts(self, QR, st_bias, mul_mode):

        # prompts = torch.cat(
        #     [
        #         st_bias,
        #         QR
        #     ],
        #     dim=1
        # )

        if mul_mode == 1:
            # simple * ： 8 N 512 * 8 1 512
            prompts = torch.mul(QR, st_bias)

        if mul_mode == 2:
            # laseElem * ： QR_bef[8 N-1 512] append (QR_last[8 1 512] * 8 1 512)
            last_element = QR[:, -1, :]
            prompts = torch.cat(
                [
                    QR[:, :-1, :],
                    last_element * st_bias

                ],
                dim = 1
            )

        if mul_mode == 3:
            # simple + ： 8 N 512 + 8 1 512
            prompts = torch.add(QR, st_bias)

        if mul_mode == 4:
            # laseElem + ： QR_bef[8 N-1 512] append (QR_last[8 1 512] + 8 1 512)
            last_element = QR[:, -1, :]
            prompts = torch.cat(
                [
                    QR[:, :-1, :],
                    last_element + st_bias

                ],
                dim=1
            )

        return prompts

    def forward(self, im_features, QR):


        wei_bias = self.meta_net(im_features)  # 8, 1, 512
        QR = QR.permute(1, 0, 2) # 8 N 512

        prompts = self.construct_prompts(QR, wei_bias, mul_mode = self.mul_op)  # same with QR

        # if self.res_flag == True:
        #     prompts = (prompts + QR) / 2

        prompts = prompts.permute(1, 0, 2) # 8 N 512

        return prompts

if __name__ == '__main__':


    im_features = torch.randn(90, 8, 512) # 8, 90, 512, inp = 90, 8, 512
    QR = torch.randn(16, 8, 512) # 8, 13, 512 , inp = N, 8, 512

    pl = PromptLearner()

    prompt = pl(im_features, QR)
    print("prompt:{}".format(prompt.shape))