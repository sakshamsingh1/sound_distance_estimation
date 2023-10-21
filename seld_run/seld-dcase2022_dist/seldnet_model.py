#
# The SELDnet architecture
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss

# define a class for calculating MSE loss for each frame + CE for frame present or not
class MSE_event_detector(object):
    def __call__(self, output, target):
        pred_dist = output[:, :, [0]]
        pred_event = output[:, :, [1]]
        loss_dist = nn.MSELoss(pred_dist, target[:, :, [0]])
        loss_event = nn.CrossEntropyLoss(pred_event, target[:, :, [1]])
        return loss_dist + loss_event

class perm_1(object):
    def __init__(self, loss_type='mse'):
        self.loss_type = loss_type
    def __call__(self, output, target):

        # mask out the frames where no event is present from output
        mask = (target != 0)
        output = output * mask

        if self.loss_type == 'mse':
            loss_func = nn.MSELoss()
        elif self.loss_type == 'mae':
            loss_func = nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}.")
        loss_dist = loss_func(output, target)
        return loss_dist

class perm_2(object):
    """
    This loss exists of two phases:
    1. only_mask: mask is learnt first
    2. mask + distance: mask and distance are jointly learnt
    """
    def __init__(self, loss_type='mse', only_mask=False, thr=0.1):
        self.only_mask = only_mask
        self.loss_type = loss_type
        self.eps = 0.1
        self.thr = thr

    def __call__(self, output, target):
        gt_mask = (target != 0).float()
        pred_mask = output[:, :, [1]]
        loss_mask = nn.BCELoss()(pred_mask, gt_mask)

        if self.only_mask:
            return loss_mask
        if self.loss_type == 'mse':
            loss_func = nn.MSELoss()

        elif self.loss_type == 'mae':
            loss_func = nn.L1Loss()

        elif self.loss_type == 'mape':
            def mape_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                return torch.mean(torch.abs((targ - pred)/targ_deno))
            loss_func = mape_loss_func

        elif self.loss_type == 'mspe':
            def mspe_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                return torch.mean(torch.square((targ - pred)/targ_deno))
            loss_func = mspe_loss_func

        elif self.loss_type == 'thr_mape':
            def thr_mape_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                per_diff = torch.abs(targ - pred)/targ_deno

                relu = nn.ReLU()
                per_diff = relu(per_diff - self.thr)
                return torch.mean(per_diff)
            loss_func = thr_mape_loss_func
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}.")

        pred_dist = output[:, :, [0]]
        pred_dist = pred_dist * pred_mask
        loss_dist = loss_func(pred_dist, target)
        return loss_mask + loss_dist

class perm_3(object):
    """
    This loss exists of two phases:
    1. only_mask: mask is learnt first
    2. mask + distance: mask and distance are jointly learnt
    """
    def __init__(self, loss_type='mse', only_mask=False, thr=0.1):
        self.only_mask = only_mask
        self.loss_type = loss_type
        self.thr = thr
        self.eps = 0.1

    def __call__(self, output, target):
        gt_mask = (target != 0).float()
        pred_mask = output[:, :, [1]]
        loss_mask = nn.BCELoss()(pred_mask, gt_mask)

        if self.only_mask:
            return loss_mask
        if self.loss_type == 'mse':
            loss_func = nn.MSELoss()

        elif self.loss_type == 'mae':
            loss_func = nn.L1Loss()

        elif self.loss_type == 'mape':
            def mape_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                return torch.mean(torch.abs((targ - pred)/targ_deno))
            loss_func = mape_loss_func

        elif self.loss_type == 'mspe':
            def mspe_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                return torch.mean(torch.square((targ - pred)/targ_deno))
            loss_func = mspe_loss_func

        elif self.loss_type == 'thr_mape':
            def thr_mape_loss_func(pred, targ):
                targ_deno = targ.clone()
                targ_deno[targ == 0] += self.eps
                per_diff = torch.abs(targ - pred)/targ_deno

                relu = nn.ReLU()
                per_diff = relu(per_diff - self.thr)
                return torch.mean(per_diff)
            loss_func = thr_mape_loss_func

        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}.")

        pred_dist = output[:, :, [0]]
        pred_dist = pred_dist * gt_mask
        loss_dist = loss_func(pred_dist, target)
        return loss_mask + loss_dist


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class CRNN(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self._onlyDist = params['only_dist']
        self._perm2 = params['permutation_2']
        self._perm3 = params['permutation_3']

        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            # out - batch x hidden x seq
            x = torch.tanh(x)

        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        if self._onlyDist:
            if self._perm2 or self._perm3:
                out = self.fnn_list[-1](x)
                out1 = nn.ReLU()(out[:, :, [0]])
                out2 = nn.Sigmoid()(out[:, :, [1]])
                doa = torch.cat((out1, out2), 2)
            else:
                doa = nn.ReLU()(self.fnn_list[-1](x))
        else:
            doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        return doa
