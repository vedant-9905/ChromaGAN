"""
create_samples.py
Creates realistic vintage B&W sample photographs for the Gradio interface presets.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

os.makedirs("assets/samples", exist_ok=True)


def create_classic_portrait():
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Vignette background
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
            val = int(np.clip(120 - dist * 0.22, 25, 120))
            img[y, x] = [val, val, val]

    # Shoulders
    cv2.ellipse(img, (w//2, int(h * 0.9)), (int(w * 0.38), int(h * 0.25)), 0, 0, 360, (50, 50, 50), -1)
    # White collar
    pts = np.array([[int(w * 0.44), int(h * 0.65)], [w//2, int(h * 0.75)], [int(w * 0.56), int(h * 0.65)]], np.int32)
    cv2.fillPoly(img, [pts], (220, 220, 220))
    # Neck & Face
    cv2.rectangle(img, (int(w * 0.44), int(h * 0.52)), (int(w * 0.56), int(h * 0.68)), (140, 140, 140), -1)
    cv2.ellipse(img, (w//2, int(h * 0.42)), (int(w * 0.18), int(h * 0.23)), 0, 0, 360, (170, 170, 170), -1)
    # Eyes
    cv2.circle(img, (int(w * 0.43), int(h * 0.40)), 8, (30, 30, 30), -1)
    cv2.circle(img, (int(w * 0.57), int(h * 0.40)), 8, (30, 30, 30), -1)
    # Lips
    cv2.ellipse(img, (w//2, int(h * 0.52)), (20, 7), 0, 0, 360, (90, 90, 90), -1)
    # Vintage Wavy Hair
    cv2.ellipse(img, (w//2, int(h * 0.30)), (int(w * 0.20), int(h * 0.12)), 0, 0, 360, (35, 35, 35), -1)

    # Film grain
    grain = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    cv2.imwrite("assets/samples/classic_portrait.png", noisy)


def create_vintage_car():
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky & Road
    img[:int(h * 0.58), :] = 160
    img[int(h * 0.58):, :] = 50

    # Distant hills
    cv2.circle(img, (int(w * 0.3), int(h * 0.65)), int(w * 0.35), (90, 90, 90), -1)
    cv2.circle(img, (int(w * 0.75), int(h * 0.65)), int(w * 0.4), (100, 100, 100), -1)

    # Car body
    cv2.rectangle(img, (int(w * 0.15), int(h * 0.44)), (int(w * 0.85), int(h * 0.64)), (60, 60, 60), -1)
    # Cabin & Windshield
    cv2.rectangle(img, (int(w * 0.35), int(h * 0.33)), (int(w * 0.68), int(h * 0.44)), (35, 35, 35), -1)
    cv2.rectangle(img, (int(w * 0.38), int(h * 0.36)), (int(w * 0.64), int(h * 0.43)), (140, 140, 140), -1)
    # Wheels
    cv2.circle(img, (int(w * 0.30), int(h * 0.64)), 50, (20, 20, 20), -1)
    cv2.circle(img, (int(w * 0.70), int(h * 0.64)), 50, (20, 20, 20), -1)
    cv2.circle(img, (int(w * 0.30), int(h * 0.64)), 22, (200, 200, 200), -1)
    cv2.circle(img, (int(w * 0.70), int(h * 0.64)), 22, (200, 200, 200), -1)

    grain = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    cv2.imwrite("assets/samples/vintage_car.png", noisy)


def create_old_city():
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky & Ground
    img[:int(h * 0.45), :] = 180
    img[int(h * 0.45):, :] = 70

    # Victorian buildings
    cv2.rectangle(img, (0, int(h * 0.15)), (int(w * 0.35), int(h * 0.80)), (85, 85, 85), -1)
    cv2.rectangle(img, (int(w * 0.65), int(h * 0.10)), (w, int(h * 0.85)), (95, 95, 95), -1)
    # Windows
    for r in range(4):
        cv2.rectangle(img, (int(w * 0.08), int(h * 0.22) + r * 60), (int(w * 0.14), int(h * 0.28) + r * 60), (230, 230, 230), -1)
        cv2.rectangle(img, (int(w * 0.20), int(h * 0.22) + r * 60), (int(w * 0.26), int(h * 0.28) + r * 60), (230, 230, 230), -1)
        cv2.rectangle(img, (int(w * 0.72), int(h * 0.18) + r * 65), (int(w * 0.78), int(h * 0.24) + r * 65), (230, 230, 230), -1)
        cv2.rectangle(img, (int(w * 0.85), int(h * 0.18) + r * 65), (int(w * 0.91), int(h * 0.24) + r * 65), (230, 230, 230), -1)

    # Street lamp
    cv2.line(img, (int(w * 0.42), int(h * 0.72)), (int(w * 0.42), int(h * 0.48)), (30, 30, 30), 4)
    cv2.circle(img, (int(w * 0.42), int(h * 0.47)), 12, (240, 240, 240), -1)

    grain = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    cv2.imwrite("assets/samples/old_city.png", noisy)


def create_yosemite_landscape():
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky & Lake
    img[:int(h * 0.5), :] = 170
    img[int(h * 0.5):, :] = 55

    # Mountains
    pts = np.array([[0, int(h * 0.5)], [int(w * 0.3), int(h * 0.18)], [int(w * 0.55), int(h * 0.42)], [int(w * 0.8), int(h * 0.15)], [w, int(h * 0.5)]], np.int32)
    cv2.fillPoly(img, [pts], (95, 95, 95))
    # Snow
    snow1 = np.array([[int(w * 0.3), int(h * 0.18)], [int(w * 0.26), int(h * 0.25)], [int(w * 0.34), int(h * 0.25)]], np.int32)
    cv2.fillPoly(img, [snow1], (240, 240, 240))

    # Pine trees
    for i in range(12):
        tx = int(i * (w / 11))
        t_pts = np.array([[tx, int(h * 0.52)], [tx + 18, int(h * 0.40)], [tx + 36, int(h * 0.52)]], np.int32)
        cv2.fillPoly(img, [t_pts], (40, 40, 40))

    grain = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    cv2.imwrite("assets/samples/yosemite_landscape.png", noisy)


if __name__ == "__main__":
    create_classic_portrait()
    create_vintage_car()
    create_old_city()
    create_yosemite_landscape()
    print("Generated 4 historical sample presets in assets/samples/")
