DISEASES = [
"Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
"Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
"Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

def encode_labels(label_string):
    vector = [0]*14
    labels = label_string.split("|")

    for l in labels:
        if l in DISEASES:
            vector[DISEASES.index(l)] = 1

    return vector