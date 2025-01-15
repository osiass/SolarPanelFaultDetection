import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models


model = models.resnet18(pretrained=False)
num_classes = 6  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('solar_panel_model.pth'))
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class_names = ['bird-drop', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']


def predict_frame(model, frame, transform, device):
    # frame: BGR (OpenCV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


cap = cv2.VideoCapture("http://192.168.1.213:8080/video")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    prediction = predict_frame(model, frame, transform, device)
    class_name = class_names[prediction]
    cv2.putText(frame, f'Prediction: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()