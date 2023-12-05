import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from images.world import create_map

# Initialize model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img

# Function to check if a Person is in the scene
def detect_person(img_path):
    img = preprocess_image(img_path)

    with torch.no_grad():
        prediction = model([img])

    for score, label in zip(
        prediction[0]['scores'], prediction[0]['labels']):
        if (label == 1 and score > 0.8 ): # label 1 is for a person
            return True
        else:
            return False

# create and print randomized trail map
trail = create_map()
print("trail map generated: ", trail, "\n\n")

# search for the hiker
i = -1
hiker_found = 0
for row in trail:
    if (hiker_found == 1):
        break
    i += 1
    j = 0
    for trail_img in row:
        if (hiker_found == 1):
            break
        if (detect_person("images/" + trail_img) == True):
            print("found hiker on map[", i, ",", j, "]")
            hiker_found = 1
        j += 1