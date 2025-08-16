import logging
import warnings

def WARNING(set='ignore'):
  warnings.filterwarnings(set)

DEVICE_NAME = 'xavier-nx'
MODEL_NAME = 'frcnn'

RUNTIME_GUARD = 0/100

EPISLON = 0.9997
MIN_EPSILON = 0
TOTAL_EPSIODES = 1000

MU = 0.1
GAMMA = 0.9

LOGGING_LEVEL = logging.INFO

if DEVICE_NAME == 'orin-nano':
    PMODE_ID = {'15W':0, '7W':1}
    MODELS = [
              {"name": "dummy", "loose": {"pb": 8000, "throughputt": 3}, "tight": {"pb": 7000, "throughputt": 4}},
            #   {"name": "efficientdet-d0", "loose": {"pb": 8000, "throughputt": 3}, "tight": {"pb": 7000, "throughputt": 3}},
            #   {"name": "fasterrcnn-mobilenetv3-large", "loose": {"pb": 8000, "throughputt": 3}, "tight": {"pb": 7000, "throughputt": 3}},
            #   {"name": "retinanet-resnet50", "loose": {"pb": 8000, "throughputt": 3}, "tight": {"pb": 7000, "throughputt": 3}},
            #   {"name": "yolov5n", "loose": {"pb": 8000, "throughputt": 3}, "tight": {"pb": 7000, "throughputt": 3}},
            ]
elif DEVICE_NAME == 'xavier-nx':
    PMODE_ID = {'MODE_20W_6CORE':8,
                'MODE_20W_4CORE':7, 'MODE_20W_2CORE':6,
                'MODE_15W_6CORE':2, 'MODE_15W_4CORE':1, 'MODE_15W_2CORE':0, 
                'MODE_10W_4CORE':4, 'MODE_10W_DESKTOP':5, 'MODE_10W_2CORE':3}
    SPEC_ORDER = {"loose":{"pb": 20000, "throughputt": 2, "scth": 65}, "tight":{"pb": 10000, "throughputt": 5, "scth": 80}}
    MODELS = [
              # {"name": "efficientdetd0"},
              {"name": "frcnn"},
              {"name": "retinanet"},
              # {"name": "yolov5n"},
    ]

CONSTRAINTS = ["tight", "loose"]
ALPHAS = [None, 0, 1, 2]