from optimum.exporters.onnx.model_configs import T5OnnxConfig, CohereOnnxConfig
from transformers import CohereConfig, AutoModelForCausalLM
from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from pathlib import Path
 
# AutoModelForCausalLM.from_pretrained("CohereForAI/aya-expanse-8b")
model_id = "CohereForAI/aya-23-8B"
config = CohereConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
#custom_cohere_onnx_config = CohereOnnxConfig(config=config, task="text-generation-with-past", use_past=True, use_past_in_inputs = True)
custom_cohere_onnx_config = CohereOnnxConfig(config=config, task="text-generation", use_past=True, use_past_in_inputs = True)
print(custom_cohere_onnx_config.DEFAULT_ONNX_OPSET)
print(custom_cohere_onnx_config.outputs)
print(custom_cohere_onnx_config.inputs)
 
onnx_path = Path("models/CohereForAI/aya-23-8B/model.onnx")
"""onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", model, task="text-generation-with-past")
onnx_config = onnx_config_constructor(model.config)
"""
print(custom_cohere_onnx_config.outputs)
print(custom_cohere_onnx_config.inputs)
 

onnx_inputs, onnx_outputs = export(model, custom_cohere_onnx_config, onnx_path, custom_cohere_onnx_config.DEFAULT_ONNX_OPSET, disable_dynamic_axes_fix=False)

print("####################################")
print("onnx_inputs")
print(onnx_inputs)
print("onnx_outputs")
print(onnx_outputs)