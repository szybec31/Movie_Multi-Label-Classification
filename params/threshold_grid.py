get_thresholds = {
    "text": {
        "distilbert": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "resnet50": [None],
        "late": [None],
    },
    "graphics": {
        "distilbert": [None],
        "resnet50": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "late": [None],
    },
    "early-fusion": {
        "distilbert": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "resnet50": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "late": [None],
    },
    "late-fusion": {
        "distilbert": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "resnet50": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        "late": [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    },
}