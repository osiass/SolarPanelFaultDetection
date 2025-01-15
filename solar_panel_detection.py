import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os

def main():
   
    # 1) Cihaz seçimi (GPU varsa CUDA, yoksa CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Transformlar (ResNet boyutu 224x224)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3) Dataset tanımlama
    train_data = datasets.ImageFolder(root='dataset/train', transform=train_transform)
    test_data  = datasets.ImageFolder(root='dataset/test',  transform=test_transform)

    # Dataloader 
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=2)

    class_names = train_data.classes  # ['bird-drop', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']
    num_classes = len(class_names)
    print("Sınıflar:", class_names)

    # 4) Model (ResNet18)
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # 5) Loss ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5  # Deneme amaçlı
    
    train_losses = []
    test_losses  = []
    train_accuracies = []
    test_accuracies  = []

    # 6) Eğitim döngüsü
    for epoch in range(num_epochs):
       
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # İstatistikler (Train)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_data)
        epoch_acc = 100.0 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
     
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(test_data)
        val_acc = 100.0 * val_correct / val_total
        
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)
        
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}% | "
              f"Test Loss: {val_loss:.4f} Test Acc: {val_acc:.2f}%")

    # 7) Eğitim & Test Grafikleri
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title('Accuracy Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 8) Son Epoch Test (Validation) Accuracy'yi veya En İyi Test Accuracy'yi göstermek
    final_test_acc = test_accuracies[-1]
    best_test_acc = max(test_accuracies)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # 9) Modeli kaydetme
    torch.save(model.state_dict(), 'solar_panel_model.pth')
    print("Model kaydedildi: solar_panel_model.pth")

    # 10) Tahmin Fonksiyonu 
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

    # 11) Video Dosyası Üzerinde Tahmin
    video_path = "birddrop.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            label_idx = predict_frame(model, frame, test_transform, device)
            label_str = class_names[label_idx]
            
            cv2.putText(frame, label_str, (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Solar Panel Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Video dosyası bulunamadı ({video_path}). Bu kısım atlanıyor...")

    # 12) Laptop Kamerası Üzerinde Tahmin
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Kamera açılamadı veya bulunamadı!")
    else:
        print("Kamera açıldı. Çıkmak için 'q'ya, tahmin almak için 's'ye basın.")
    
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kamera okunamadı!")
                break
            
            cv2.imshow("Laptop Kamerasi - Solar Panel", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                label_idx = predict_frame(model, frame, test_transform, device)
                label_str = class_names[label_idx]
                print("Tahmin Edilen Sınıf:", label_str)

        cap.release()
        cv2.destroyAllWindows()

# Windows'ta multiprocessing hatasını çözmek için
if __name__ == '__main__':
    main()
