import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)
def load_image(image_name):
  image = Image.open(image_name).convert("RGB")
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

original_img = load_image("owl.png")
style_img = load_image("style.jpg")

# complete new function
def imshow(tensor, title=None):
  image = tensor.cpu().clone().squeeze(0)
  image = transforms.ToPILImage()(image)
  plt.imshow(image)
  if title:
    plt.title(title)
  plt.pause(0.001)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalization mean and std
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class ContentLoss(nn.Module):
  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    self.loss = nn.functional.mse_loss(input, self.target)
    return input

def gram_matrix(input):
  batch_size, channels, height, width = input.size()
  features = input.view(channels, height*width)
  G = torch.mm(features, features.t())
  return G.div(channels * height * width)

class StyleLoss(nn.Module):
  def __init__ (self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
    G = gram_matrix(input)
    self.loss = nn.functional.mse_loss(G, self.target)
    return input

    
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, original_img):
  normalization = nn.Sequential(transforms.Normalize(normalization_mean, normalization_std))

  content_losses = []
  style_losses = []

  model = nn.Sequential(normalization)

  i=0
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = f'conv_{i}'
    elif isinstance(layer, nn.ReLU):
      name = f'relu_{i}'
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
      name = f'pool{i}'
    elif isinstance(layer, nn.BatchNorm2d):
      name = f'bn_{i}'
    else:
      raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
    
    model.add_module(name, layer)

    if name in content_layers:
      target = model(original_img).detach()
      content_loss = ContentLoss(target)
      model.add_module(f'content_loss_{i}', content_loss)
      content_losses.append(content_loss)

    if name in style_layers:
      target_feature = model(style_img).detach()
      style_loss = StyleLoss(target_feature)
      model.add_module(f'style_loss_{i}', style_loss)
      style_losses.append(style_loss)

    
  for i in range(len(model) -1, -1, -1):
    if isinstance(model[i], (ContentLoss, StyleLoss)):
      break
  model = model[:(i + 1)]

  return model, style_losses, content_losses

input_img = original_img.clone()
input_img.requires_grad_(True)

optimizer = optim.LBFGS([input_img])

model, style_losses, content_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, original_img)

run=[0]
while run[0] <= 6000:
  def closure():
    input_img.data.clamp_(0, 1)

    optimizer.zero_grad()
    model(input_img)
    style_score = sum(sl.loss for sl in style_losses)
    content_score = sum(cl.loss for cl in content_losses)

    loss = style_score * 1e6 + content_score *1
    loss.backward()

    run[0] += 1
    if run[0] % 50 == 0:
      print(f'Run {run[0]}:')
      print(f'Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}')
      imshow(input_img, title=f'step {run[0]}')

    return style_score + content_score
  
  optimizer.step(closure)

input_img.data.clamp_(0, 1)
imshow(input_img, title='Output Image') 
final_image = input_img.cpu().clone().squeeze(0)
final_image = transforms.ToPILImage()(final_image)
final_image.save("stylized_output.png")