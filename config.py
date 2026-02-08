import torch

class Config:
    # Dataset
    DATASET_NAME = "CIC-IDS-2017"
    DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
    NUM_FEATURES = 78
    NUM_CLASSES = 14
    ATTACK_CLASSES = [
        "BENIGN", "Bot", "DDoS", "DDoS_GoldenEye", "DoS_Hulk",
        "DoS_Slowhttptest", "DoS_Slowloris", "FTP-Patator",
        "PortScan", "SSH-Patator", "Web_Attack_Brute_Force",
        "Web_Attack_XSS"
    ]
    
    # Model
    MODEL_TYPE = "CNN"  # CNN, ResNet, Autoencoder
    HIDDEN_DIM = 128
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Federated Learning
    NUM_NODES = 11
    FEDERATED_ROUNDS = 20
    FEDAVG = True
    
    # TabFIDS
    USE_BOOTSTRAPPING = True
    USE_TEMPORAL_AVERAGING = True
    TEMPORAL_WINDOW = 3
    
    # BBSA
    USE_BBSA = True
    BLOCK_SIZE = 4
    
    # DDFE
    USE_DDFE = False
    FEATURE_REDUCTION_RATIO = 0.6
    
    # Paths
    DATA_DIR = "./data"
    MODELS_DIR = "./models"
    RESULTS_DIR = "./results"