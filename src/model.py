from models.generic_cv.EfficientNetV2_S.model import EfficientNetV2
# from models.generic_cv.ConvNeXt-V2.model import ConvNeXtV2
# from models.generic_cv.Swin_Transformer.model import SwinTransformer
from models.medical_sota.RadJEPA.model import RadJEPA
from config import NUM_CLASSES, MODEL_NAME

def get_model():
    if MODEL_NAME == "efficientnet":
        return EfficientNetV2(NUM_CLASSES)

    elif MODEL_NAME == "convnext":
        return ConvNeXtV2(NUM_CLASSES)

    elif MODEL_NAME == "swin":
        return SwinTransformer(NUM_CLASSES)
    
    elif MODEL_NAME == "radjepa":
        return RadJEPA(NUM_CLASSES)

    else:
        raise ValueError("Unknown model")