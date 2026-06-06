from ultralytics import YOLO

def main():
    # اگر اسم فایل مدل شما چیز دیگری است، همینجا اصلاح کنید
    model = YOLO(r'E:\project\models\yolov8s.pt')

    model.train(
        data='quadrant.yaml',
        epochs=50,
        imgsz=693,
        batch=4
    )

if __name__ == '__main__':
    main()
