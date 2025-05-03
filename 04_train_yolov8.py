from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='D:\\Projects\\Guvi_Project5\\SolarGuard\\data\\final_yolo_dataset\\data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='solar_guard_detector',
    project='runs/detect',
    device='cpu'
)
