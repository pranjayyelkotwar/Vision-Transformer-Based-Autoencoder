import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig
import timm

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196, num_heads=8, num_layers=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.to_pixel_values = nn.Linear(embed_dim, 16 * 16 * 3)  # Map back to pixel space

    def forward(self, x):
        x = x + self.pos_embed  # Add positional encoding
        x = self.decoder(x, x)  # Transformer decoding
        x = self.to_pixel_values(x)  # Convert to image space
        x = x.view(x.size(0), 14, 14, 16, 16, 3)  # Reshape to (B, 14, 14, 16, 16, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # Rearrange to (B, 224, 224, 3)
        return x

class ViTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.decoder = TransformerDecoder()
    
    def forward(self, x):
        x = self.encoder(x).last_hidden_state[:, 1:, :]  # Ignore CLS token
        x = self.decoder(x)  # Decode into image
        return x

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = ViTAutoencoder().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4)

# Example Training Step
def train_step(images):
    images = images.to(device)
    optimizer.zero_grad()
    recon_images = autoencoder(images)
    loss = criterion(recon_images, images)
    loss.backward()
    optimizer.step()
    return loss.item()
