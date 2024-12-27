# Real-Time Urban Object Detection System with TensorRT Optimization

A high-performance object detection system optimized for urban environments, capable of detecting vehicles, pedestrians, traffic lights, and traffic signs in real-time using YOLO architecture and TensorRT optimization.

## Features

- Real-time object detection using YOLOv11l architecture
- TensorRT optimization for improved inference speed
- Support for both image and video processing
- Web-based interface using Streamlit
- Multi-GPU training support
- Batch processing for videos
- Custom preprocessing pipeline for optimal input handling
- Performance metrics monitoring

## Requirements

### Hardware Requirements

- CUDA-capable GPU (Recommended: NVIDIA GPU with at least 8GB VRAM)
- Minimum 16GB RAM
- SSD storage for faster data processing

### Software Requirements

- Python 3.8+
- CUDA Toolkit 11.x
- cuDNN 8.x
- TensorRT 8.x
- FFmpeg

### Python Dependencies

Install dependencies using pip:

```bash
pip install roboflow ultralytics torch torchvision tensorrt streamlit opencv-python pillow numpy
```

## Dataset

The project uses the BDD100K dataset, which has been preprocessed and is available on Roboflow:

- Dataset URL: [BDD100K on Roboflow](https://universe.roboflow.com/pedro-azevedo-3c9ol/bdd100k-3zgda/dataset/5)
- Classes: car, person, traffic light, traffic sign
- Version: 5
- Format: YOLOv11 compatible

## Training Process

The model must be trained before using it in the web application. Training can be done using the provided Jupyter notebook (`train.ipynb`) in one of the following environments:

### Option 1: Google Colab (Recommended for free GPU access)

1. Upload `train.ipynb` to Google Colab
2. Add your Roboflow API key in the notebook
3. Use GPU runtime (Go to Runtime > Change runtime type > GPU)
4. Run all cells in sequence
5. Download the trained weights from the `runs/detect/train/weights/` directory

### Option 2: Kaggle

1. Create a new notebook or upload `train.ipynb`
2. Enable GPU accelerator (Settings > Accelerator > GPU)
3. Add your Roboflow API key
4. Run all cells in sequence
5. Download the trained weights

### Option 3: Local PC (For users with powerful GPUs)

1. Install required packages:

```bash
pip install roboflow ultralytics torch torchvision
```

2. Run the notebook in Jupyter Lab or Jupyter Notebook
3. Make sure to use GPU acceleration

### Training Configuration

The notebook includes optimized training parameters:

- Image size: 1280x1280
- Batch size: 8 (adjust based on your GPU memory)
- Epochs: 30
- AdamW optimizer with cosine learning rate scheduling
- Advanced augmentations (mixup, mosaic)
- Early stopping and model checkpointing

After training, the weights will be saved in the `runs/detect/train/weights/` directory.

## Deployment

### Converting to TensorRT

After training, convert your PyTorch model to TensorRT for faster inference:

```bash
python engine_creator.py
```

### Running the Web Interface

1. Start the Streamlit application:

```bash
streamlit run webapp.py
```

2. Access the interface at `http://localhost:8501`

### Web Interface Features

- Upload images or videos for processing
- Adjust confidence threshold in real-time
- View inference time and GPU usage metrics
- Download processed media with annotations
- Batch processing for video files
- Real-time visualization of detection results

## Project Structure

```
urban-object-detection/
├── webapp.py           # Streamlit web application
├── engine_creator.py   # TensorRT conversion script
├── train.ipynb         # Training notebook for Colab/Kaggle
└── README.md          # This file
```

## Performance Metrics

Expected performance after training and TensorRT optimization:

- Real-time inference (30+ FPS on supported hardware)
- Batch processing capability for videos
- GPU memory optimization
- FP16 precision for improved performance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

Common issues and solutions:

1. Out of Memory during training:

   - Reduce batch size
   - Reduce image size
   - Use gradient accumulation

2. TensorRT conversion fails:

   - Ensure compatible TensorRT version
   - Check CUDA toolkit version
   - Verify GPU compatibility

3. Low inference speed:
   - Enable GPU acceleration
   - Verify TensorRT optimization
   - Check batch size settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the GitHub repository.
