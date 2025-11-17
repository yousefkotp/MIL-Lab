from src.models import abmil, transmil, transformer, ilra, dftd, clam, rrt, wikg, dsmil, meanmil
import pathlib
import os
REPO_PATH = str(pathlib.Path(__file__).parent.resolve())  # absolute path to repo root
CONFIG_PATH = os.path.join(REPO_PATH, 'model_configs')
MODEL_SAVE_PATH = os.path.join(REPO_PATH, 'model_weights')

ENCODER_DIM_MAPPING : dict[str, int] = {
    'uni': 1024,
    'uni_v1': 1024,
    'uni_v2': 1536,
    'ctranspath': 768,
    'conch': 512,
    'conch_v1': 512,
    'conch_v15': 768,
    'gigapath': 1536,
    'resnet50': 2048,
    'virchow': 2560,
    'virchow2': 2560,
    'phikon': 768,
    'phikon_v1': 768,
    'phikon_v2': 1024,
    'phikon2': 1024,
    'hoptimus': 1536,
    'hoptimus_0': 1536,
    'hoptimus_1': 1536,
    'hoptimus1': 1536,
    'musk': 1024,
    'slide_hubert': 1024,
    'slide_hubert_base': 768,
    'gigapath': 1536,
    'dino_vit_small_p8': 384,
    'dino_vit_small_p8_embeddings': 384,
    'hibou_b': 768,
    'hibou_l': 1024,
    'midnight12k': 1536,
    'concat_1_head': 256,
    'concat_2_heads': 512,
    'concat_3_heads': 768,
    'concat_4_heads': 1024,
}

MODEL_ENTRYPOINTS = {
    'meanmil': (meanmil.MeanMILModel, meanmil.MeanMILConfig),
    'abmil': (abmil.ABMILModel, abmil.ABMILGatedBaseConfig),
    'transmil': (transmil.TransMILModel, transmil.TransMILConfig),
    'transformer': (transformer.TransformerModel, transformer.TransformerConfig),
    'dftd': (dftd.DFTDModel, dftd.DFTDConfig),
    'clam': (clam.CLAMModel, clam.CLAMConfig),
    'ilra': (ilra.ILRAModel, ilra.ILRAConfig),
    'rrt': (rrt.RRTMILModel, rrt.RRTMILConfig),
    'wikg': (wikg.WIKGMILModel, wikg.WIKGConfig),
    'dsmil': (dsmil.DSMILModel, dsmil.DSMILConfig),
}  
