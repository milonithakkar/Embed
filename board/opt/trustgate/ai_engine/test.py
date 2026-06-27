import openvino as ov

core = ov.Core()
print(core.available_devices)
# Example output: ['CPU', 'GPU', 'NPU']
