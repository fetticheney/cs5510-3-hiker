import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms

# Initialize the model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img

# Preprocess the image
img_path = "images/sec1.jpg"
img = preprocess_image(img_path)

# Perform the prediction
with torch.no_grad():
    prediction = model([img])

print("Prediction: ", prediction)

# Function to extract labels and scores
def get_person_labels(prediction):
    labels = []
    scores = prediction[0]['scores']
    for score, label in zip(scores, prediction[0]['labels']):
        if label == 1:  # assuming 1 is the label for 'person'
            labels.append(f'Person: {score:.2f}')
    return labels

# Function to extract bounding boxes
def get_person_boxes(prediction, threshold=0.5):
    boxes = []
    for element, score in enumerate(prediction[0]['scores']):
        if prediction[0]['labels'][element] == 1 and score > threshold: # Label for 'person'
            boxes.append(prediction[0]['boxes'][element].tolist())
    return boxes

# Get labels and boxes
labels = get_person_labels(prediction)
person_boxes = get_person_boxes(prediction)

print("Labels: ", labels)
print("Person Boxes: ", person_boxes)

# Function to draw bounding boxes and labels
def draw_boxes_with_labels(img_path, boxes, labels):
    img = Image.open(img_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    if not boxes:
        print("No boxes to draw.")
        return

    if len(boxes) != len(labels):
        print("The number of boxes and labels don't match.")
        return

    for box, label in zip(boxes, labels):
        if len(box) == 4:
            x, y, x2, y2 = box
            print(f"Drawing box: {box}, Label: {label}")
            rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x2, y2, label, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        else:
            print(f"Invalid box dimensions: {box}")

    plt.show()

# Call the draw function
draw_boxes_with_labels(img_path, person_boxes, labels)
