
FIELD_WIDTH = 20
FIELD_HEIGHT = 20
FIELD_DIM = 2

N_HIDDEN = 10
FC_H_SIZE = 256

EPOCHS = 600

BATCH_SIZE = 20
LEARNING_RATE = 1e-5
INEQ_DEPTH = 100
EQ_DEPTH = 100

TEST_SIZE = 500
RENDER_INDEX = 8

MODEL_TYPE = "optnet"
#MODEL_TYPE = "fc"

#ACTION = "generate"
#ACTION = "train"
ACTION = "render"

PATHS = {
    "magnets-train": "data/it2/magnets-500-train.npy",
    "fields-train": "data/it2/fields-500-train.npy",
    "magnets-test": "data/it2/magnets-500-test.npy",
    "fields-test": "data/it2/fields-500-test.npy",
    "save-results": "data/results/500-it6" + MODEL_TYPE + ".npy",
    "model": "models/final-" + MODEL_TYPE + ".pt"
}

