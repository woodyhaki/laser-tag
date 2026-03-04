import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.convdim = nn.Conv2d(in_channels, out_channels, 
                                     kernel_size=1, stride=stride)
        else:
            self.convdim = None
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.bn1(self.conv1(out))
        
        if self.convdim is not None:
            identity = self.convdim(identity)
            
        out += identity
        return self.relu(out)

class SimpleImageEncoder(nn.Module):
    def __init__(self, depth, width):
        super(SimpleImageEncoder, self).__init__()
        self.image_encoder = SimpleResNetEncoder(depth, width)

    def forward(self, image):
        batch_size, seq_len, channel, height, width = image.shape
        image_input = image.reshape((batch_size * seq_len, channel, height, width))
        # patch_embedding, g0, g1, g2 = self.image_encoder(image_input)
        patch_embedding = self.image_encoder(image_input)
        
        patch_embedding = patch_embedding.reshape((batch_size, seq_len, -1))
        return patch_embedding

class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super(SimpleCNNEncoder, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(64, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        return x

class SimpleResNetEncoder(nn.Module):
    def __init__(self, depth, width):
        super(SimpleResNetEncoder, self).__init__()
        
        # Compute depth n of each group and adjust width
        n = (depth - 2) // 6   # For depth=32, n=5
        
        widths = [int(v * width) for v in (16, 32, 64)]
        
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.gn = nn.GroupNorm(num_groups=4, num_channels=16)
        self.relu = nn.ReLU(inplace=True)
        
        self.group0 = self._make_group('group0', 16, widths[0], n)
        self.group1 = self._make_group('group1', widths[0], widths[1], n)
        self.group2 = self._make_group('group2', widths[1], widths[2], n)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(widths[2], num_classes)
        self.fc = nn.Identity()
        
        
    def _make_group(self, name, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(Block(in_channels, out_channels))
            else:
                layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    # def forward(self, x):
    #     # Initial convolution
    #     x = self.conv0(x)
    #     x = self.gn(x)
    #     x = self.relu(x)
        
    #     # Pass through group0
    #     g0 = self.group0(x)
        
    #     # Pass through group1
    #     g1 = self.group1(g0)
        
    #     # Pass through group2
    #     g2 = self.group2(g1)
        
    #     # Average pooling and fully connected
    #     x = self.avgpool(g2)
        
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #     # print(f"x shape:{x.shape}")
        
    #     return x, g0, g1, g2
    
    def forward(self, x):
        # Initial convolution
        x = self.conv0(x)
        x = self.gn(x)
        x = self.relu(x)
        
        # Pass through group0
        g0 = self.group0(x)
        
        # Pass through group1
        g1 = self.group1(g0)
        
        # Pass through group2
        g2 = self.group2(g1)
        
        # Average pooling and fully connected
        x = self.avgpool(g2)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # print(f"x shape:{x.shape}")
        
        return x