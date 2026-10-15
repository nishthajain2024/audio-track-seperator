"""
Module H: Embedded Deployment Preparation
Converts TensorFlow 1D U-Net checkpoint into optimized TensorFlow Lite (.tflite) models
for low-latency, memory-efficient inference on embedded ARM Linux devices (Raspberry Pi).
"""

import os
import sys
import numpy as np

# Ensure project root in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from Model.UNetModel import AudioTrackSeparation


def convert_unet_to_tflite(
    weights_path: str = "./models/model_weights",
    output_dir: str = "./embedded_deployment",
    dim: int = 75008
):
    os.makedirs(output_dir, exist_ok=True)
    print("================================================================================")
    print("      1D U-Net TensorFlow Lite (TFLite) Embedded Model Converter")
    print("================================================================================")
    print(f"[*] Input Weights : {weights_path}")
    print(f"[*] Target Dim    : {dim} samples (~6.80s at 11,025 Hz)")
    print(f"[*] Output Dir    : {output_dir}")
    print("--------------------------------------------------------------------------------")

    # 1. Instantiate and build model
    model = AudioTrackSeparation()
    model.build(input_shape=(None, dim, 1))

    # Resolve weights
    if os.path.isdir(weights_path):
        cand = os.path.join(weights_path, 'model_weights')
        if os.path.exists(cand + '.index') or os.path.exists(cand):
            weights_path = cand
    elif not (os.path.exists(weights_path + '.index') or os.path.exists(weights_path)):
        alt = os.path.join(project_root, weights_path)
        if os.path.isdir(alt):
            cand = os.path.join(alt, 'model_weights')
            if os.path.exists(cand + '.index'):
                weights_path = cand
        elif os.path.exists(alt + '.index'):
            weights_path = alt

    print(f"[+] Restoring checkpoint from: {weights_path}")
    model.load_weights(weights_path).expect_partial()

    # 2. Convert to Concrete Function with fixed batch dimension for embedded execution
    run_model = tf.function(lambda x: model(x, training=False))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, dim, 1], dtype=tf.float32, name="input_audio")
    )

    # 3. Standard Float32 TFLite Model
    print("[+] Exporting Standard Float32 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model_fp32 = converter.convert()

    fp32_path = os.path.join(output_dir, "unet_audio_separator_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_model_fp32)
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"    --> Saved: {fp32_path} ({fp32_size_mb:.2f} MB)")

    # 4. Dynamic-Range Quantized TFLite Model (Optimal for ARM Cortex-A)
    print("[+] Exporting Dynamic-Range Quantized (INT8 weights) TFLite model...")
    converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    try:
        tflite_model_quant = converter_quant.convert()
        quant_path = os.path.join(output_dir, "unet_audio_separator_quant.tflite")
        with open(quant_path, "wb") as f:
            f.write(tflite_model_quant)
        quant_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
        print(f"    --> Saved: {quant_path} ({quant_size_mb:.2f} MB)")
        compression_ratio = (1.0 - (quant_size_mb / fp32_size_mb)) * 100
        print(f"    --> Footprint Reduction: {compression_ratio:.1f}% smaller!")
    except Exception as e:
        print(f"    [!] Quantization note: {str(e)}")

    print("================================================================================")
    print("TFLite export finished successfully!")


if __name__ == '__main__':
    convert_unet_to_tflite()
