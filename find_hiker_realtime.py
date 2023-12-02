import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    image = transform(image)
    return image

# Function to extract bounding boxes
def get_person_boxes(prediction, threshold=0.5):
    boxes = []
    for element, score in enumerate(prediction[0]['scores']):
        if prediction[0]['labels'][element] == 1 and score > threshold: # Label for 'person'
            boxes.append(prediction[0]['boxes'][element].tolist())
    return boxes

# Function to extract labels and scores
def get_person_labels(prediction):
    labels = []
    scores = prediction[0]['scores']
    for score, label in zip(scores, prediction[0]['labels']):
        if label == 1:  # assuming 1 is the label for 'person'
            labels.append(f'Person: {score:.2f}')
    return labels

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image
    img = preprocess_image(frame)

    # Perform the prediction
    with torch.no_grad():
        prediction = model([img])

    # Get labels and boxes
    labels = get_person_labels(prediction)
    person_boxes = get_person_boxes(prediction)

    # Draw bounding boxes and labels on the frame
    for box, label in zip(person_boxes, labels):
        if len(box) == 4:
            x, y, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
