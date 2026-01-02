import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# load_interpreter loads a TFLite interpreter from a model file
def load_interpreter(model_path: str) -> tf.lite.Interpreter:
    """
    Load a TFLite interpreter from a model file.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# tflite_fp32_predict performs inference using a TFLite interpreter
def tflite_fp32_predict(
    interpreter: tf.lite.Interpreter, image: np.ndarray
) -> tuple[int, float]:
    """
    Perform inference using a TFLite interpreter.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    return np.argmax(output[0]), np.max(output[0])


# load_single_image loads a single image and preprocesses it
def load_single_image(img_path: str, target_size=(128, 128)) -> np.ndarray:
    """
    Loads and preprocesses a single image exactly like ImageDataGenerator
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array


def __main__():
    parser = argparse.ArgumentParser(
        description="Rice Grain Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="../models/tflite/rice_classifier.tflite",
        help="Path to TFLite model (default: models/tflite/rice_classifier.tflite)",
    )

    parser.add_argument("image", type=str, help="Path to input image")

    args = parser.parse_args()

    TFLITE_FP32_PATH = args.model
    Image_PATH = args.image

    if not os.path.exists(TFLITE_FP32_PATH):
        print(f"Model not found: {TFLITE_FP32_PATH}")
        sys.exit(1)

    if not os.path.exists(Image_PATH):
        print(f"Image not found: {Image_PATH}")
        sys.exit(1)

    try:
        classes = ["Karacadag", "Ipsala", "Arborio", "Basmati", "Jasmine"]

        tflite_fp32 = load_interpreter(TFLITE_FP32_PATH)

        pred, confidence = tflite_fp32_predict(
            tflite_fp32, load_single_image(Image_PATH)
        )

        print("\n" + "=" * 60)
        print("RICE GRAIN CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"Image: {Image_PATH}")
        print("-" * 60)
        print(f"Prediction: {classes[pred]} ({confidence*100:.2f}% confidence)")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    __main__()
