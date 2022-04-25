def convert_float_to_float16(model_path: str, output_model_path: str):
    from onnxmltools.utils.float16_converter import (
        convert_float_to_float16_model_path,
    )

    new_onnx_model = convert_float_to_float16_model_path(model_path)

    onnx.save(new_onnx_model, output_model_path)

    # Alternate approach
    # from onnx import load_model
    # from onnxruntime.transformers import optimizer, onnx_model
    #
    # # optimized_model = optimizer.optimize_model(model_path, model_type='bert')
    #
    # model = load_model(model_path)
    # optimized_model = onnx_model.OnnxModel(model)
    #
    # if hasattr(optimized_model, 'convert_float32_to_float16'):
    #     optimized_model.convert_float_to_float16()
    # else:
    #     optimized_model.convert_model_float32_to_float16()
    #
    # self._textual_path = f'{self._textual_path[:-5]}_optimized.onnx'
    # optimized_model.save_model_to_file(output_model_path)


def quantize(model_path: str, output_model_path: str):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_model_path,
        per_channel=True,
        reduce_range=True,  # should be the same as per_channel
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
        optimize_model=True,
        op_types_to_quantize=["MatMul", "Attention", "Mul", "Add"],
        extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True},
    )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
