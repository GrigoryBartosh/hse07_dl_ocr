import torch
import torch.nn as nn

from common.utils import activation_by_name

__all__ = ['Stacker']


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1,
        stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=7,
        stride=stride, padding=3, bias=False
    )


def upsample3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=stride),
        conv3x3(in_planes, out_planes)
    )


def upsample7x7(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=stride),
        conv7x7(in_planes, out_planes)
    )


class SimpleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(SimpleBlock, self).__init__()

        if out_planes is None:
            out_planes = in_planes * self.expansion

        conv = upsample3x3 if upsample_block else conv3x3

        self.conv1 = conv(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.activ1 = activation_by_name(activ)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ1(out)

        return out


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(ResBasicBlock, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        self.residual = None
        if stride != 1 or in_planes != out_planes:
            self.residual = nn.Sequential(
                (upsample3x3 if upsample_block else conv1x1)(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

        conv = upsample3x3 if upsample_block else conv3x3

        self.activ = activation_by_name(activ)

        self.conv1 = conv(in_planes, planes, 1 if upsample_block else stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, out_planes, stride if upsample_block else 1)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual is not None:
            identity = self.residual(x)

        out += identity
        out = self.activ(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(ResBottleneck, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        self.residual = None
        if stride != 1 or in_planes != out_planes:
            self.residual = nn.Sequential(
                (upsample3x3 if upsample_block else conv1x1)(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

        conv = upsample3x3 if upsample_block else conv3x3

        self.activ = activation_by_name(activ)

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual is not None:
            identity = self.residual(x)

        out += identity
        out = self.activ(out)

        return out


class Encoder(nn.Module):
    def __init__(self, block, layer_sizes, activ='relu'):
        super(Encoder, self).__init__()

        in_planes = 64
        self.conv1 = conv7x7(1, in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activ1 = activation_by_name(activ)
        self.conv2 = conv3x3(in_planes, in_planes, stride=2)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.activ2 = activation_by_name(activ)

        layers = [self._make_layer(block, in_planes, in_planes, 1, layer_sizes[0], activ)]
        for layer_size in layer_sizes[1:]:
            layers += [self._make_layer(block, in_planes, in_planes * 2, 2, layer_size, activ)]
            in_planes = in_planes * 2

        self.layer = nn.Sequential(*layers)

    def _make_layer(self, block, in_planes, out_planes, stride, layer_size, activ):
        layers = [block(
            in_planes, out_planes=out_planes,
            stride=stride, activ=activ
        )]

        for _ in range(1, layer_size):
            layers += [block(out_planes, activ=activ)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activ2(x)

        x = self.layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, image_size, block, layer_sizes, in_planes=512, start_planes=512, activ='lrelu'):
        super(Decoder, self).__init__()

        self.image_size = image_size

        layers = [self._make_layer(block, in_planes, start_planes // 2, 2, layer_sizes[0], activ)]
        in_planes = start_planes // 2
        for layer_size in layer_sizes[1:-1]:
            layers += [self._make_layer(block, in_planes, in_planes // 2, 2, layer_size, activ)]
            in_planes = in_planes // 2
        layers += [self._make_layer(block, in_planes, in_planes, 1, layer_sizes[-1], activ)]
        self.layer = nn.Sequential(*layers)

        self.conv1 = upsample3x3(in_planes, out_planes=64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.activ1 = activation_by_name(activ)
        self.conv2 = upsample7x7(in_planes=64, out_planes=1, stride=2)
        self.activ2 = nn.Tanh()

    def _make_layer(self, block, in_planes, out_planes, stride, layer_size, activ):
        layers = []
        for _ in range(layer_size - 1):
            layers += [block(in_planes, activ=activ, upsample_block=True)]

        layers += [block(
            in_planes=in_planes, out_planes=out_planes,
            stride=stride, activ=activ, upsample_block=True
        )]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)

        x = self.conv2(x)
        x = self.activ2(x)

        pad = (x.size(2) - self.image_size) // 2
        x = x[:, :, pad:-pad, pad:-pad]

        return x


class Mover(nn.Module):
    def __init__(self, args):
        super(Mover, self).__init__()

        layer_sizes = args['encoder']['layers']
        self.encoder = Encoder(
            self.get_block_by_name(args['encoder']['block']),
            layer_sizes,
            args['encoder']['activ']
        )

        planes = 64 * 2 ** (len(layer_sizes) - 1)

        self.decoder = Decoder(
            args['decoder']['image_size'],
            self.get_block_by_name(args['decoder']['block']),
            args['decoder']['layers'],
            planes + args['params_move_count'],
            planes,
            args['decoder']['activ']
        )

    def get_block_by_name(self, block_name):
        if block_name == 'SimpleBlock':
            return SimpleBlock
        elif block_name == 'ResBasicBlock':
            return ResBasicBlock
        elif block_name == 'ResBottleneck':
            return ResBottleneck

        assert False, f"Unsupported block: {args['block']}"

    def forward(self, x, x_params_move):
        x = x[:, None, :, :]

        x = self.encoder(x)

        x_params_move = x_params_move[:, :, None, None]
        x_params_move = x_params_move.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, x_params_move), 1)

        x = self.decoder(x)

        x = x.squeeze(1)

        return x


class Stacker(nn.Module):
    def __init__(self, args):
        super(Stacker, self).__init__()

        self.mover = Mover(args)

    def move(self, x_t, x_params_move):
        return self.mover(x_t, x_params_move)

    def stack(self, x_i, x_t, x_rgb):
        x_t = (x_t + 1) / 2
        x_t = x_t[:, None, :, :]
        x_rgb = x_rgb[:, :, None, None]
        return x_i * (1 - x_t) + x_rgb * x_t

    def forward(self, x_i, x_t, x_params):
        x_params_move, x_rgb = x_params[:, :5], x_params[:, 5:]
        x_t = self.move(x_t, x_params_move)
        return self.stack(x_i, x_t, x_rgb)