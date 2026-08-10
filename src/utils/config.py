# config.py

WANTED_CHANNELS = ['PZ', 'PO3', 'PO4', 'PO5', 'PO6', 'POZ', 'OZ', 'O1', 'O2']
NUM_CHANNELS = len(WANTED_CHANNELS)

WINDOW_SIZE = 512
WINDOW_STRIDE = 128

BATCH_SIZE = 32

LATENT_DIM = 24
NOISE_DIM = LATENT_DIM  # noise dimension equals latent dim; label info is injected via embeddings

# Class conditioning
NUM_CLASSES = 15
ALL_STIM_FREQS = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0]
LABEL_EMB_DIM = 16

# Sinusoidal frequency conditioning (replaces label embedding in G, S, D)
SSVEP_FS = 1000            # EEG sampling frequency in Hz
FREQ_N_HARMONICS = 3       # fundamental + 2nd + 3rd harmonic
FREQ_DIM = 2 * FREQ_N_HARMONICS  # sin+cos per harmonic → 6 channels

LR_EMBEDDER = 5e-4
LR_RECOVERY = 1e-3
EPOCHS_EMBEDDER = 500
NUM_LAYERS_RECOVERY = 2
NUM_LAYERS_EMBEDDER = 2

LR_SUPERVISOR = 1e-3
LR_SUPERVISOR_JOINT = 1e-4
EPOCHS_SUPERVISOR = 200
NUM_LAYERS_SUPERVISOR = 2

LR_GENERATOR = 1e-4
LR_DISCRIMINATOR = 8e-5
EPOCHS_JOINT = 250
WARMUP_EPOCHS = 10   # supervised-only pre-training epochs for G before adversarial training
HIDDEN_DIM_DISCRIMINATOR = 128
HIDDEN_DIM_GENERATOR = 128
NUM_LAYERS_GENERATOR = 4
NUM_LAYERS_DISCRIMINATOR = 4

LAMBDA_SUP = 20.0
LAMBDA_MOM = 2.2
LAMBDA_SPEC = 1.5
LAMBDA_ADV = 1.0

D_STEPS_PER_G_STEP = 5
R1_PENALTY_WEIGHT = 10.0

LR_CLASSIFIER = 1e-3
EPOCHS_CLASSIFIER = 50