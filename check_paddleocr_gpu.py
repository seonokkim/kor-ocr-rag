from paddleocr import PaddleOCR
import cv2
import os
import paddle

# Check PaddlePaddle GPU availability
gpu_available = paddle.device.is_compiled_with_cuda()
print(f"PaddlePaddle compiled with CUDA: {gpu_available}")

if gpu_available:
    # Specify the path to the image file
    image_path = 'data/test/images/5350224/1994/5350224-1994_000001.jpg' # Using one of the test images

    # Initialize PaddleOCR with GPU enabled
    # The 'use_gpu' parameter is expected to work based on documentation
    # Use lang='korean' for Korean language support
    try:
        print("Initializing PaddleOCR with GPU...")
        # use_gpu=True 대신 use_gpu=1 사용
        ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=1)
        print("PaddleOCR initialized successfully.")

        # Perform OCR on the image
        if os.path.exists(image_path):
            result = ocr.ocr(image_path, cls=True)

            # Print the detected text and bounding boxes
            if result is not None:
                print("\nOCR Results:")
                for line in result:
                    for word_info in line:
                        bbox = word_info[0]
                        text = word_info[1][0]
                        confidence = word_info[1][1]
                        print(f"Text: {text}, BBox: {bbox}, Confidence: {confidence:.4f}")
            else:
                print("No text detected.")
        else:
            print(f"Error: Image file not found at {image_path}")

    except Exception as e:
        print(f"An error occurred during PaddleOCR initialization or OCR: {e}")

else:
    print("GPU is not available or PaddlePaddle is not compiled with CUDA. Skipping PaddleOCR GPU test.") 