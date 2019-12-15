import torch
import torch.nn as nn
import torchvision.models as models

from common.utils import activation_by_name

__all__ = ['Generator']


class ImageEncoder(nn.Module):
    def __init__(self, model_name, emb_size, pretrained=False):
        super(ImageEncoder, self).__init__()

        if model_name[:3] == 'vgg':
            if model_name == 'vgg16':
                model_type = models.vgg16_bn
            elif model_name == 'vgg19':
                model_type = models.vgg19_bn
            else:
                assert False, f"Unsupported model: {args['model']}"

            vgg = model_type(pretrained=pretrained)
            vgg_children = list(vgg.children())
            self.model = nn.Sequential(*vgg_children[:-1])
            self.head = nn.Sequential(
                *list(vgg_children[-1].children())[:-1],
                nn.Linear(4096, emb_size)
            )
        elif model_name[:6] == 'resnet':
            if model_name == 'resnet18':
                model_type = models.resnet18
            elif model_name == 'resnet34':
                model_type = models.resnet34
            elif model_name == 'resnet50':
                model_type = models.resnet50
            elif model_name == 'resnet152':
                model_type = models.resnet152
            else:
                assert False, f"Unsupported model: {model_name}"

            self.model = model_type(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            if model_name in ['resnet18', 'resnet34']:
                self.head = nn.Linear(512, emb_size)
            else:
                self.head = nn.Linear(2048, emb_size)
        else:
            assert False, f"Unsupported model: {model_name}"

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x


class MLP(nn.Module):
    def __init__(self, layer_dims, activ='relu', dropout_rate=0):
        super(MLP, self).__init__()

        layers = []
        in_dim, layer_dims = layer_dims[0], layer_dims[1:]
        for dim in layer_dims[:-1]:
            layers += [
                nn.Linear(in_dim, dim),
                activation_by_name(activ),
                nn.Dropout(dropout_rate)
            ]
            in_dim = dim

        layers += [nn.Linear(in_dim, layer_dims[-1])]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.model_image = ImageEncoder(
            args['model_image'],
            args['emb_size'],
            args['model_image_pretrained']
        )
        self.model_text = ImageEncoder(
            args['model_text'],
            args['emb_size'],
            args['model_text_pretrained']
        )

        layer_dims = args['mlp_layers'][:-1]
        self.params_count = layer_dims[-1]
        layer_dims += [self.params_count * 2]
        self.mlp = MLP(
            layer_dims,
            args['activ'],
            args['dropout_rate'],
        )
        self.tanh = nn.Tanh()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x_i, x_t):
        x_i = self.model_image(x_i)
        x_t = self.model_image(x_t)

        x = torch.cat((x_i, x_t), 1)
        x = self.mlp(x)

        mu, logvar = x[:self.params_count], x[self.params_count:]
        x = self.reparameterize(mu, logvar)
        x = self.tanh(x)

        return x