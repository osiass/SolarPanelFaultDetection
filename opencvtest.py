# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:18:31 2025

@author: lenovo
"""

import cv2

url = "http://RASPPI IP:PORT/?action=stream"

# Canlı yayına bağlan
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamıyor!")
        break

    cv2.imshow("Raspberry Pi Kamera Yayını", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
