from models.generic_cv.EfficientNetV2_S.model import EfficientNetV2
from models.generic_cv.ConvNeXt_V2.model import ConvNeXtV2
from models.generic_cv.Swin_Transformer.model import SwinTransformer
from models.medical_sota.RadJEPA.model import RadJEPA
from models.medical_sota.RadDINO.model import RadDINO
from config import NUM_CLASSES, MODEL_NAME

def get_model():
    if MODEL_NAME == "efficientnet":
        return EfficientNetV2(NUM_CLASSES)

    elif MODEL_NAME == "convnext":
        return ConvNeXtV2(NUM_CLASSES)

    elif MODEL_NAME == "swin":
        return SwinTransformer(NUM_CLASSES)

    elif MODEL_NAME == "raddino":
        return RadDINO(NUM_CLASSES, freeze_backbone=True)
    
    elif MODEL_NAME == "radjepa":
        return RadJEPA(NUM_CLASSES)

    elif MODEL_NAME == "resnet50":
        return ResNet50(NUM_CLASSES)

    else:
        raise ValueError("Unknown model")