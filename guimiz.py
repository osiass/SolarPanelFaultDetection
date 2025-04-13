#gps webcam gui
import cv2
import torch
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import torchvision.models as models
import threading
import time
import os
import random
import csv
from datetime import datetime
import pynmea2  # NMEA cümlelerini parse etmek için
import serial    # Seri port iletişimi için
import asyncio
import nest_asyncio
import websockets

# Websocket için nest_asyncio uygulanıyor
nest_asyncio.apply()

# Global variables
model = None
device = None
transform = None
class_names = ['bird-drop', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']
cap = None
is_running = False
current_video_path = ""
current_coordinates = None
detections = []
captured_frames = {}
model_error = False
websocket_connection = None

# Raspberry Pi websocket bağlantı bilgileri
RASPBERRY_IP = "RASPPI IP"
PORT = 8765

# Global UI elements
root = None
video_label = None
source_label = None
detection_label = None
prediction_var = None
lat_var = None
lon_var = None
alt_var = None
time_var = None
log_text = None
issues_count_var = None
start_button = None
stop_button = None
export_button = None
file_path_var = None
camera_url_var = None
export_folder_var = None
detection_thread = None
ws_delay_var = None  # GPS veri gecikme süresi için

def load_model():
    """Load the solar panel detection model"""
    global model, model_error, device
    
    model = models.resnet18(pretrained=False)
    num_classes = 6
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    try:
        model.load_state_dict(torch.load('solar_panel_model.pth', map_location=torch.device('cpu')))
        print("Model loaded successfully")
        model_error = False
    except Exception as e:
        print(f"Error loading model: {e}")
        model_error = True
        
    model.eval()
    return model

def get_simulated_coordinates():
    """Simulate drone GPS coordinates (fallback)"""
    base_lat = 40.7128
    base_lon = -74.0060
    base_alt = 50.0
    lat = base_lat + random.uniform(-0.0005, 0.0005)
    lon = base_lon + random.uniform(-0.0005, 0.0005)
    alt = base_alt + random.uniform(-2.0, 2.0)
    
    return {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "altitude": round(alt, 1),
        "timestamp": time.strftime("%H:%M:%S")
    }

def update_coordinate_display():
    """Update the coordinate UI elements with current coordinates"""
    global current_coordinates, lat_var, lon_var, alt_var, time_var
    if current_coordinates:
        coords = current_coordinates
        lat_var.set(f"{coords['latitude']}")
        lon_var.set(f"{coords['longitude']}")
        alt_var.set(f"{coords['altitude']} m")
        time_var.set(f"{coords['timestamp']}")

def log_detection(class_name, coords):
    """Add detection to the log with timestamp and coordinates"""
    global log_text, detections, issues_count_var, captured_frames
    
    if class_name != 'clean':
        formatted_name = class_name.replace('_', ' ').title()
        log_entry = f"[{coords['timestamp']}] {formatted_name} detected at: " \
                    f"Lat: {coords['latitude']}, Lon: {coords['longitude']}, Alt: {coords['altitude']}m\n"
        log_text.insert(tk.END, log_entry)
        log_text.see(tk.END)
        
        detection = {
            'class': class_name,
            'timestamp': coords['timestamp'],
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'altitude': coords['altitude']
        }
        detections.append(detection)
        issues_count_var.set(str(len(detections)))

def browse_file():
    global file_path_var, camera_url_var
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )
    if file_path:
        file_path_var.set(file_path)
        camera_url_var.set("")

def browse_export_folder():
    global export_folder_var
    folder_path = filedialog.askdirectory()
    if folder_path:
        export_folder_var.set(folder_path)

def predict_frame(frame):
    global model, device, transform, class_names
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def run_detection():
    global is_running, cap, current_coordinates, prediction_var, captured_frames
    frames_to_update_coords = 10
    frame_count = 0
    frame_number = 0
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            stop_detection()
            break
        frame_number += 1
        frame_count += 1
        # Güncel GPS verisi update ediliyorsa onu kullan; yoksa simüle et
        if frame_count >= frames_to_update_coords:
            if current_coordinates is None:
                current_coordinates = get_simulated_coordinates()
            frame_count = 0
        
        update_coordinate_display()
        
        try:
            prediction = predict_frame(frame)
            class_name = class_names[prediction]
            formatted_class_name = class_name.replace('_', ' ').title()
            prediction_var.set(formatted_class_name)
            
            if class_name != 'clean':
                log_detection(class_name, current_coordinates)
                if class_name not in captured_frames:
                    captured_frames[class_name] = frame.copy()
            
            coords = current_coordinates
            cv2.putText(frame, f'Prediction: {formatted_class_name}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Lat: {coords["latitude"]}, Lon: {coords["longitude"]}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Alt: {coords["altitude"]}m - Time: {coords["timestamp"]}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Frame: {frame_number}', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Prediction error: {e}")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        video_frame_width = video_label.winfo_width()
        video_frame_height = video_label.winfo_height()
        width, height = img.size
        display_width = video_frame_width - 20
        display_height = video_frame_height - 20
        if display_width > 10 and display_height > 10:
            ratio = min(display_width/width, display_height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img = img_tk
        video_label.config(image=img_tk)
        time.sleep(0.03)

def get_correct_camera_url():
    url = camera_url_var.get().strip()
    if url.endswith("stream.html"):
        url = url.replace("stream.html", "?action=stream")
    return url

def start_detection():
    global model_error, file_path_var, camera_url_var, cap, is_running
    global current_video_path, detections, captured_frames, source_label
    global detection_label, start_button, stop_button, export_button
    global issues_count_var, log_text, detection_thread, current_coordinates
    
    if model_error:
        messagebox.showerror("Model Error", 
            "Model file 'solar_panel_model.pth' not found. Please ensure the model file is in the same directory.")
        return
        
    file_path = file_path_var.get().strip()
    camera_url = camera_url_var.get().strip()
    
    if file_path:
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        source = file_path
        source_label.config(text=f"File: {os.path.basename(file_path)}")
    elif camera_url:
        source = get_correct_camera_url()
        source_label.config(text=f"Camera: {source}")
    else:
        messagebox.showinfo("Info", "Please select a video file or enter a camera URL")
        return
    
    detections.clear()
    captured_frames.clear()
    issues_count_var.set("0")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source")
        return
    
    current_video_path = source
    is_running = True
    detection_label.config(text="Running")
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    export_button.config(state=tk.DISABLED)
    log_text.delete(1.0, tk.END)
    
    # Eğer gerçek GPS verisi okunamıyorsa simülasyon kullanılır
    if current_coordinates is None:
        current_coordinates = get_simulated_coordinates()
    
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()

def stop_detection():
    global is_running, cap, detection_label, start_button, stop_button
    global prediction_var, export_button, detections
    is_running = False
    if cap:
        cap.release()
    detection_label.config(text="Stopped")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    prediction_var.set("None")
    
    if len(detections) > 0:
        export_button.config(state=tk.NORMAL)
        messagebox.showinfo("Detection Complete", 
            f"Processing complete! Found {len(detections)} issues.\n"
            f"Click 'Export CSV' to save the results.")
    else:
        messagebox.showinfo("Detection Complete", "Processing complete! No issues found.")

def ensure_export_folder():
    global export_folder_var
    export_folder = export_folder_var.get()
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    images_folder = os.path.join(export_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    return export_folder, images_folder

def export_csv():
    global detections, captured_frames
    if not detections:
        messagebox.showinfo("No Data", "No issues were detected to export.")
        return
    export_folder, images_folder = ensure_export_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(export_folder, f"solar_panel_issues_{timestamp}.csv")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['class', 'timestamp', 'latitude', 'longitude', 'altitude', 'image_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, detection in enumerate(detections):
                class_name = detection['class']
                if class_name in captured_frames:
                    img_filename = f"{class_name}_{timestamp}.jpg"
                    img_path = os.path.join(images_folder, img_filename)
                    if not os.path.exists(img_path):
                        cv2.imwrite(img_path, captured_frames[class_name])
                    detection_with_img = detection.copy()
                    detection_with_img['image_path'] = os.path.join("images", img_filename)
                    writer.writerow(detection_with_img)
                    if class_name in captured_frames:
                        del captured_frames[class_name]
                else:
                    detection_with_img = detection.copy()
                    detection_with_img['image_path'] = ''
                    writer.writerow(detection_with_img)
        messagebox.showinfo("Export Successful", 
            f"Data exported successfully to:\n{csv_filename}\n\n"
            f"Images saved to:\n{images_folder}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Error exporting data: {e}")

async def process_gps_data(data):
    """
    NMEA mesajlarını işleyerek koordinatlara dönüştürür
    """
    global current_coordinates
    try:
        if data.startswith("$GPGGA"):
            msg = pynmea2.parse(data)
            # gps_qual 0 ise fix alınmamış demektir.
            if hasattr(msg, 'gps_qual') and int(msg.gps_qual) > 0 and hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                current_coordinates = {
                    "latitude": msg.latitude,
                    "longitude": msg.longitude,
                    "altitude": float(msg.altitude) if hasattr(msg, 'altitude') and msg.altitude else 0.0,
                    "timestamp": msg.timestamp.strftime("%H:%M:%S") if hasattr(msg, 'timestamp') and msg.timestamp else time.strftime("%H:%M:%S")
                }
                print(f"GPS güncellendi: {current_coordinates}")
                return True
        return False
    except Exception as e:
        print(f"GPS veri işleme hatası: {e}")
        return False

async def receive_gps_websocket():
    """
    Raspberry Pi'den WebSocket üzerinden GPS verilerini alır
    """
    global current_coordinates, RASPBERRY_IP, PORT, ws_delay_var
    
    while True:
        try:
            uri = f"ws://{RASPBERRY_IP}:{PORT}"
            print(f"GPS websocket bağlantısı kuruluyor: {uri}")
            
            async with websockets.connect(uri) as websocket:
                print("GPS verileri alınıyor...")
                
                while True:
                    try:
                        # WebSocket'ten veri al
                        data = await websocket.recv()
                        print(f"GPS ham veri: {data}")
                        
                        # Veriyi işle
                        success = await process_gps_data(data)
                        
                        # İşleme başarısızsa simüle et
                        if not success and current_coordinates is None:
                            current_coordinates = get_simulated_coordinates()
                        
                        # Gecikmeli alım için bekle
                        delay = int(ws_delay_var.get()) if ws_delay_var.get().isdigit() else 0
                        if delay > 0:
                            await asyncio.sleep(delay)
                            
                    except Exception as e:
                        print(f"GPS veri alma hatası: {e}")
                        break
                        
        except Exception as conn_err:
            print(f"WebSocket bağlantı hatası: {conn_err}")
            # Bağlantı hatası durumunda simüle et
            current_coordinates = get_simulated_coordinates()
            await asyncio.sleep(5)  # Yeniden bağlanmadan önce bekle

def start_gps_thread():
    """
    GPS verileri için asyncio event loop'unu başlatır
    """
    async def run_gps_loop():
        await receive_gps_websocket()
    
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_gps_loop())
    
    gps_thread = threading.Thread(target=run_async_loop)
    gps_thread.daemon = True
    gps_thread.start()

def update_raspberry_connection():
    """
    Raspberry Pi bağlantı ayarlarını güncelle
    """
    global RASPBERRY_IP, PORT, raspberry_ip_var, raspberry_port_var
    RASPBERRY_IP = raspberry_ip_var.get().strip()
    PORT = int(raspberry_port_var.get().strip()) if raspberry_port_var.get().strip().isdigit() else 8765
    messagebox.showinfo("Bağlantı Güncellendi", 
        f"Raspberry Pi bağlantı ayarları güncellendi:\nIP: {RASPBERRY_IP}\nPort: {PORT}")

def create_ui():
    global root, video_label, source_label, detection_label, prediction_var
    global lat_var, lon_var, alt_var, time_var, log_text, issues_count_var
    global start_button, stop_button, export_button, file_path_var
    global camera_url_var, export_folder_var, model_error, class_names
    global raspberry_ip_var, raspberry_port_var, ws_delay_var
    
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    control_canvas_frame = ttk.Frame(main_frame)
    control_canvas_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    
    control_canvas = tk.Canvas(control_canvas_frame, width=300)
    control_scrollbar = ttk.Scrollbar(control_canvas_frame, orient="vertical", command=control_canvas.yview)
    
    control_frame_container = ttk.Frame(control_canvas)
    control_canvas.create_window((0, 0), window=control_frame_container, anchor="nw")
    
    control_frame = ttk.LabelFrame(control_frame_container, text="Controls", padding=10)
    control_frame.pack(fill=tk.BOTH, expand=True)
    
    # Raspberry Pi WebSocket Bağlantı Ayarları
    raspberry_frame = ttk.LabelFrame(control_frame, text="Raspberry Pi GPS Bağlantısı", padding=10)
    raspberry_frame.pack(fill=tk.X, pady=10)
    
    raspberry_ip_frame = ttk.Frame(raspberry_frame)
    raspberry_ip_frame.pack(fill=tk.X, pady=2)
    ttk.Label(raspberry_ip_frame, text="IP:", width=8).pack(side=tk.LEFT)
    raspberry_ip_var = tk.StringVar(value=RASPBERRY_IP)
    ttk.Entry(raspberry_ip_frame, textvariable=raspberry_ip_var, width=15).pack(side=tk.LEFT, padx=5)
    
    raspberry_port_frame = ttk.Frame(raspberry_frame)
    raspberry_port_frame.pack(fill=tk.X, pady=2)
    ttk.Label(raspberry_port_frame, text="Port:", width=8).pack(side=tk.LEFT)
    raspberry_port_var = tk.StringVar(value=str(PORT))
    ttk.Entry(raspberry_port_frame, textvariable=raspberry_port_var, width=15).pack(side=tk.LEFT, padx=5)
    
    delay_frame = ttk.Frame(raspberry_frame)
    delay_frame.pack(fill=tk.X, pady=2)
    ttk.Label(delay_frame, text="Gecikme (s):", width=8).pack(side=tk.LEFT)
    ws_delay_var = tk.StringVar(value="0")
    ttk.Entry(delay_frame, textvariable=ws_delay_var, width=5).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(raspberry_frame, text="Bağlantıyı Güncelle", command=update_raspberry_connection).pack(pady=5)
    
    ttk.Label(control_frame, text="Video Source:").pack(anchor=tk.W, pady=5)
    
    file_frame = ttk.Frame(control_frame)
    file_frame.pack(fill=tk.X, pady=5)
    file_path_var = tk.StringVar()
    ttk.Entry(file_frame, textvariable=file_path_var, width=20).pack(side=tk.LEFT, padx=5)
    ttk.Button(file_frame, text="Browse", command=browse_file).pack(side=tk.LEFT)
    
    camera_frame = ttk.Frame(control_frame)
    camera_frame.pack(fill=tk.X, pady=5)
    ttk.Label(camera_frame, text="or Camera URL:").pack(side=tk.LEFT)
    camera_url_var = tk.StringVar()
    ttk.Entry(camera_frame, textvariable=camera_url_var, width=20).pack(side=tk.LEFT, padx=5)
    
    export_frame = ttk.Frame(control_frame)
    export_frame.pack(fill=tk.X, pady=5)
    ttk.Label(export_frame, text="Export Folder:").pack(anchor=tk.W)
    folder_frame = ttk.Frame(export_frame)
    folder_frame.pack(fill=tk.X, pady=2)
    export_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "detections"))
    ttk.Entry(folder_frame, textvariable=export_folder_var, width=20).pack(side=tk.LEFT, padx=5)
    ttk.Button(folder_frame, text="Browse", command=browse_export_folder).pack(side=tk.LEFT)
    
    button_frame = ttk.Frame(control_frame)
    button_frame.pack(fill=tk.X, pady=10)
    start_button = ttk.Button(button_frame, text="Start", command=start_detection)
    start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(button_frame, text="Stop", command=stop_detection, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=5)
    export_button = ttk.Button(button_frame, text="Export CSV", command=export_csv, state=tk.DISABLED)
    export_button.pack(side=tk.LEFT, padx=5)
    
    status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
    status_frame.pack(fill=tk.X, pady=10)
    ttk.Label(status_frame, text="Current Source:").pack(anchor=tk.W)
    source_label = ttk.Label(status_frame, text="None")
    source_label.pack(anchor=tk.W, padx=10)
    ttk.Label(status_frame, text="Detection:").pack(anchor=tk.W, pady=(10, 0))
    detection_label = ttk.Label(status_frame, text="Not Running")
    detection_label.pack(anchor=tk.W, padx=10)
    ttk.Label(status_frame, text="Issues Found:").pack(anchor=tk.W, pady=(10, 0))
    issues_count_var = tk.StringVar(value="0")
    ttk.Label(status_frame, textvariable=issues_count_var).pack(anchor=tk.W, padx=10)
    
    prediction_frame = ttk.LabelFrame(control_frame, text="Current Prediction", padding=10)
    prediction_frame.pack(fill=tk.X, pady=10)
    prediction_var = tk.StringVar(value="None")
    ttk.Label(prediction_frame, textvariable=prediction_var, font=("Arial", 14, "bold")).pack(pady=10)
    
    coordinate_frame = ttk.LabelFrame(control_frame, text="Drone Coordinates", padding=10)
    coordinate_frame.pack(fill=tk.X, pady=10)
    lat_frame = ttk.Frame(coordinate_frame)
    lat_frame.pack(fill=tk.X, pady=2)
    ttk.Label(lat_frame, text="Latitude:", width=10).pack(side=tk.LEFT)
    lat_var = tk.StringVar(value="--")
    ttk.Label(lat_frame, textvariable=lat_var).pack(side=tk.LEFT)
    lon_frame = ttk.Frame(coordinate_frame)
    lon_frame.pack(fill=tk.X, pady=2)
    ttk.Label(lon_frame, text="Longitude:", width=10).pack(side=tk.LEFT)
    lon_var = tk.StringVar(value="--")
    ttk.Label(lon_frame, textvariable=lon_var).pack(side=tk.LEFT)
    alt_frame = ttk.Frame(coordinate_frame)
    alt_frame.pack(fill=tk.X, pady=2)
    ttk.Label(alt_frame, text="Altitude:", width=10).pack(side=tk.LEFT)
    alt_var = tk.StringVar(value="--")
    ttk.Label(alt_frame, textvariable=alt_var).pack(side=tk.LEFT)
    time_frame = ttk.Frame(coordinate_frame)
    time_frame.pack(fill=tk.X, pady=2)
    ttk.Label(time_frame, text="Time:", width=10).pack(side=tk.LEFT)
    time_var = tk.StringVar(value="--")
    ttk.Label(time_frame, textvariable=time_var).pack(side=tk.LEFT)
    
    log_frame = ttk.LabelFrame(control_frame, text="Detected Issues Log", padding=10)
    log_frame.pack(fill=tk.X, pady=10)
    log_text = tk.Text(log_frame, height=6, width=30, wrap=tk.WORD)
    log_scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    legend_frame = ttk.LabelFrame(control_frame, text="Prediction Classes", padding=10)
    legend_frame.pack(fill=tk.X, pady=10)
    for name in class_names:
        ttk.Label(legend_frame, text=f"• {name.replace('_', ' ').title()}").pack(anchor=tk.W)
    
    control_frame_container.update_idletasks()
    control_canvas.config(scrollregion=control_frame_container.bbox("all"), yscrollcommand=control_scrollbar.set)
    control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def on_frame_configure(event):
        control_canvas.configure(scrollregion=control_canvas.bbox("all"))
    control_frame_container.bind("<Configure>", on_frame_configure)
    
    def on_mouse_wheel(event):
        control_canvas.yview_scroll(-1 * (event.delta // 120), "units")
    control_canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    
    video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding=10)
    video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    video_label = ttk.Label(video_frame)
    video_label.pack(fill=tk.BOTH, expand=True)
    
    if model_error:
        error_label = ttk.Label(
            video_frame, 
            text="Error: Model file 'solar_panel_model.pth' not found.\nPlease ensure the model file is in the same directory as this script.",
            foreground="red",
            font=("Arial", 12)
        )
        error_label.pack(pady=20)

def initialize_app():
    global root, model, device, transform, current_coordinates
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Başlangıç koordinatlarını simülasyonla ata
    current_coordinates = get_simulated_coordinates()
    create_ui()
    
    # GPS WebSocket bağlantısını başlat
    start_gps_thread()

def main():
    global root
    root = tk.Tk()
    root.title("Solar Panel Damage Detection with GPS Coordinates")
    root.geometry("1000x700")  # Arayüz boyutunu biraz büyüttüm
    initialize_app()
    root.mainloop()

if __name__ == "__main__":
    main()