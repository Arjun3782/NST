import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size
image_size = 356

# Image loader
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

def load_image(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load images
original_img = load_image("owl.png")
style_img = load_image("style.jpg")

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# VGG19 model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalization
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Content loss module
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Gram matrix for style
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Style loss module
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Layers to use
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Model builder
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = nn.Sequential(
        nn.BatchNorm2d(3),  # Add dummy normalization (torchvision uses it differently)
    )
    
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim unused layers
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

# Clone input
input_img = original_img.clone()
input_img.requires_grad_(True)

# Get model and losses
model, style_losses, content_losses = get_style_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std, style_img, original_img
)

# Adam optimizer
optimizer = optim.Adam([input_img], lr=0.01)

# Train
num_steps = 600
style_losses_history = []
content_losses_history = []
steps = []

for step in range(1, num_steps + 1):
    input_img.data.clamp_(0, 1)

    optimizer.zero_grad()
    model(input_img)

    style_score = sum(sl.loss for sl in style_losses)
    content_score = sum(cl.loss for cl in content_losses)
    loss = style_score * 1e6 + content_score

    loss.backward()
    optimizer.step()

    style_losses_history.append(style_score.item())
    content_losses_history.append(content_score.item())
    steps.append(step)

    if step % 50 == 0:
        print(f"Step {step}: Style Loss = {style_score.item():.4f}, Content Loss = {content_score.item():.4f}")
        imshow(input_img, title=f"Step {step}")

# Final result
input_img.data.clamp_(0, 1)
imshow(input_img, title='Final Output')
final_image = input_img.cpu().clone().squeeze(0)
final_image = transforms.ToPILImage()(final_image)
final_image.save("stylized_output_adam.png")

# Plot loss curves
plt.figure()
plt.plot(steps, style_losses_history, label='Style Loss')
plt.plot(steps, content_losses_history, label='Content Loss')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")
plt.grid(True)
plt.savefig("loss_curves.png")
plt.show()
