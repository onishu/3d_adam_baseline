def build_model(name: str):
    name = name.lower()
    if name == "patchcore":
        from anomalib.models import Patchcore
        return Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True, coreset_sampling_ratio=0.1, num_neighbors=9)
    if name == "padim":
        from anomalib.models import Padim
        return Padim(backbone="resnet18", layers=("layer1", "layer2", "layer3"), pre_trained=True)
    if name == "cfa":
        from anomalib.models import Cfa
        return Cfa(backbone="wide_resnet50_2", gamma_c=1, gamma_d=1, num_nearest_neighbors=3, num_hard_negative_features=3, radius=1e-5)
    raise ValueError(f"Unknown model: {name}")

MAX_EPOCHS = {"patchcore": 1, "padim": 1, "cfa": 10}
