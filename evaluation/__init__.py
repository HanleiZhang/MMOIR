from .score_func.energy import func as ENERGY
from .score_func.ma import func as MA
from .score_func.vim import func as VIM
from .score_func.maxlogit import func as MAXLOGIT
from .score_func.msp import func as MSP
from .score_func.residual import func as RESIDUAL

ood_detection_map = {
    'energy': ENERGY,
    'ma': MA,
    'vim': VIM,
    'maxlogit': MAXLOGIT,
    'msp': MSP,
    'residual': RESIDUAL
}