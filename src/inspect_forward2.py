import sys
sys.path.insert(0, '..')
import config
from src.model import get_model
import inspect

config.MODEL_NAME = "radjepa"
m = get_model()
attn_layer = m.encoder.model.blocks[0].attn
print(inspect.getsource(attn_layer.forward))
