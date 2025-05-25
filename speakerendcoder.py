import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import  ResNet18_Weights
import torchvision.models as models

class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()
        
        # Use a pre-trained ResNet as the base model
        # ResNet works well with spectrograms which are 2D representations like images
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first layer to accept single-channel input (mel spectrogram)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a new fully connected layer for speaker embedding
        self.embedding_layer = nn.Linear(512, embedding_dim)
        
        # Classification layer for training
        self.classifier = nn.Linear(embedding_dim, num_speakers)
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # Generate embeddings
        embeddings = self.embedding_layer(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Classification logits
        logits = self.classifier(embeddings)
        
        return embeddings, logits
    
    def extract_embedding(self, x):
        """Use this method for inference to only get the embedding"""
        with torch.no_grad():
            features = self.features(x)
            features = torch.flatten(features, 1)
            embeddings = self.embedding_layer(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings