import asyncio
import serial
import websockets

GPS_PORT = "/dev/ttyAMA0"  # GPS modülünün bağlı olduğu port
GPS_BAUDRATE = 9600

async def gps_handler(websocket, path):
    print("Bir istemci bağlandı.")
    try:
        with serial.Serial(GPS_PORT, GPS_BAUDRATE, timeout=1) as gps_serial:
            while True:
                line = gps_serial.readline().decode('ascii', errors='replace').strip()
                if line:
                    await websocket.send(line)
                await asyncio.sleep(0.1)  # gereksiz CPU yükünü azaltmak için
    except Exception as e:
        print("GPS okuma/gönderme hatası:", e)

start_server = websockets.serve(gps_handler, "0.0.0.0", 8765)

print("WebSocket sunucusu başlatıldı: ws://0.0.0.0:8765")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()