import torch.nn as nn
from torchvision import models as models_2d
import timm
import torch


def load_partial_state_dict(model, checkpoint_path):
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 获取模型的权重
    model_dict = model.state_dict()

    # 过滤出与模型匹配的权重
    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

    # 打印匹配的和未匹配的权重
    print("Matched keys:", len(matched_state_dict.keys()))
    # print("Unmatched keys from checkpoint:", set(state_dict.keys()) - set(matched_state_dict.keys()))

    # 加载匹配的权重到模型
    model_dict.update(matched_state_dict)
    model.load_state_dict(model_dict)

    return model



class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models_2d.resnet18(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True):
    model = models_2d.resnet50(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50_spark(pretrained=True):
    model= timm.models.resnet50(pretrained=False)
    if pretrained:
        model = load_partial_state_dict(model, '/haoranlai/Project/CARZero/PretrainModel/resnet/resnet50_1kpretrained_timm_style.pth')
    feature_dims = model.fc.in_features
    model.fc = Identity()

    return model, feature_dims, 1024

################################################################################
# DenseNet Family
################################################################################


def densenet_121(pretrained=True):
    model = models_2d.densenet121(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_161(pretrained=True):
    model = models_2d.densenet161(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_169(pretrained=True):
    model = models_2d.densenet169(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


################################################################################
# ResNextNet Family
################################################################################


def resnext_50(pretrained=True):
    model = models_2d.resnext50_32x4d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None


def resnext_100(pretrained=True):
    model = models_2d.resnext101_32x8d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None
