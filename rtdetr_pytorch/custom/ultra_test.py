from ultralytics import YOLO
from ultralytics import NAS
from time import time

# YOLO
# model = YOLO('yolov5s.pt')  # pretrained YOLOv8n model
model = NAS('yolo_nas_s.pt')

print('Evaluating YOLO...')
results = model(['bus.jpg'])  # return a list of Results objects

time0 = time()
results = model(['bus.jpg'])  # return a list of Results objects
print(f'YOLO inference time：{(time() - time0) * 1000:.1f}ms')

# Process results list
results[0].save('result0.jpg')
