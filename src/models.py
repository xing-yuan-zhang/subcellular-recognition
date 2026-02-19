import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_outputs: int = 1, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_outputs)
        return m

    if model_name == "vgg16":
        m = models.vgg16(pretrained=pretrained)

        in_features = m.classifier[-1].in_features
        classifier = list(m.classifier.children())
        classifier[-1] = nn.Linear(in_features, num_outputs)
        m.classifier = nn.Sequential(*classifier)
        return m

    if model_name == "densenet121":
        m = models.densenet121(pretrained=pretrained)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, num_outputs)
        return m

    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=pretrained)

        if isinstance(m.classifier, nn.Sequential):
            in_features = m.classifier[-1].in_features
            new_classifier = list(m.classifier.children())
            new_classifier[-1] = nn.Linear(in_features, num_outputs)
            m.classifier = nn.Sequential(*new_classifier)
        else:
            in_features = m.classifier.in_features
            m.classifier = nn.Linear(in_features, num_outputs)
        return m

    raise ValueError(f"Unsupported model_name: {model_name}")
