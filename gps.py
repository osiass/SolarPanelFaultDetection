import socket
import pynmea2

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("192.168.1.200", 5555))  # Raspberry Pi’nin IP adresini gir

while True:
    data = client.recv(1024).decode().strip()
    if data.startswith("$GPGGA") or data.startswith("$GPRMC"):  # GPS verisi olup olmadığını kontrol et
        try:
            msg = pynmea2.parse(data)
            print(f"Enlem: {msg.latitude}, Boylam: {msg.longitude}")
        except pynmea2.ParseError:
            print(f"Hatalı veri: {data}")
