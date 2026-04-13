from models.generic_cv.EfficientNet_B0.model import EfficientNet_B0
from models.generic_cv.ConvNeXt_V2.model import ConvNeXtV2
from models.generic_cv.Swin_Transformer.model import SwinTransformer
from models.medical_sota.RadJEPA.model import RadJEPA
from models.medical_sota.RadDINO.model import RadDINO
from models.baseline.Custom_Model.model import CNNTransformerFromScratch

from models.baseline.ResNet50.model import ResNet50
import config

def get_model():
    if config.MODEL_NAME == "efficientnet":
        return EfficientNet_B0(config.NUM_CLASSES)

    elif config.MODEL_NAME == "convnext":
        return ConvNeXtV2(config.NUM_CLASSES)

    elif config.MODEL_NAME == "swin":
        return SwinTransformer(config.NUM_CLASSES)

    elif config.MODEL_NAME == "raddino":
        return RadDINO(config.NUM_CLASSES, freeze_backbone=True)
    
    elif config.MODEL_NAME == "radjepa":
        return RadJEPA(config.NUM_CLASSES)
    
    elif config.MODEL_NAME == "cnn_transformer":
        return CNNTransformerFromScratch(config.NUM_CLASSES)

    elif config.MODEL_NAME == "resnet50":
        return ResNet50(config.NUM_CLASSES)

    else:
        raise ValueError("Unknown model")