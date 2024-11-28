from optimum.exporters.onnx.model_configs import T5OnnxConfig, CohereOnnxConfig
from transformers import CohereConfig, AutoModelForCausalLM
from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from pathlib import Path

# AutoModelForCausalLM.from_pretrained("CohereForAI/aya-expanse-8b")
model_id = "CohereForAI/aya-expanse-8b"
config = CohereConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
custom_cohere_onnx_config = CohereOnnxConfig(config=config, task="text-generation", use_past=True)
print(custom_cohere_onnx_config.DEFAULT_ONNX_OPSET)
# print(custom_cohere_onnx_config.outputs)
print(custom_cohere_onnx_config.inputs)


onnx_path = Path("custom_aya-expanse-8b_onnx/model.onnx")
onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", model, task="text-generation-with-past")
onnx_config = onnx_config_constructor(model.config)
onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)
