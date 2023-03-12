import lightly.data as data

# the collate function applies random transforms to the input images
collate_fn = data.ImageCollateFunction(input_size=32, cj_prob=0.5)
import torch

# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir='./my/cute/cats/dataset/')

# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=128,         # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn)  # apply transformations to the input images
import torchvision

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

# use a resnet backbone
resnet = torchvision.models.resnext101_64x4d()  # or efficientnet0_b0
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# build a SimCLR model
class SimCLR(torch.nn.Module):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

model = SimCLR(resnet, hidden_dim=512, out_dim=128)

# use a criterion for self-supervised learning
# (normalized temperature-scaled cross entropy loss)
criterion = NTXentLoss(temperature=0.5)

# get a PyTorch optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_epochs = 10
for epoch in range(max_epochs):
    for (x0, x1), _, _ in dataloader:

        x0 = x0.to(device)
        x1 = x1.to(device)

        z0 = model(x0)
        z1 = model(x1)

        loss = criterion(z0, z1)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()