import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/student_model_int8.tflite")
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

print("=== TENSOR DETAILS ===")
for d in details:
    if any(k in d['name'] for k in ['conv', 'dense', 'output', 'ecg_input']):
        try:
            tensor = interpreter.get_tensor(d['index'])
            print(f"{d['name']} | shape={tensor.shape} | type={d['dtype']} | quant={d.get('quantization_parameters')}")
        except Exception:
            print(f"{d['name']} | shape={d['shape']} | type={d['dtype']} | quant={d.get('quantization_parameters')}")
