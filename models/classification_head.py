import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.classfy = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classfy(x.view(x.size(0), -1))