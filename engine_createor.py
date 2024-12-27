from ultralytics import YOLO

# Load your PT model
model = YOLO("D:/1Work/Projects/Object Detection/Urban Object Detection/final train results/weights/best.pt")

# Export to TensorRT engine
model.export(format="engine", 
            device=0,  # Use GPU
            half=True,  # Use FP16 for better performance, reduces memory usage and increases inference speed
            simplify=True,
            workspace=4,  # Workspace size in GB, 4 means 4GB of GPU memory can be used during optimization
            verbose=True)  # Show conversion progress