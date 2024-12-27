from .MAG_BERT import MAG_BERT
from .MULT import MULT
from .TCL_MAP import TCL_MAP
from .MMIM import MMIM
from .SDIF import SDIF
from .CC import CCModel
from .MCN import MCNModel
from .UMC import UMCModel
from .USNID import USNIDModel
from .SCCL import SCCLModel

multimodal_methods_map = {
    'mag_bert': MAG_BERT,
    'mult': MULT,
    'mmim': MMIM,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
    'usnid': USNIDModel,
    'mcn': MCNModel,
    'cc': CCModel,
    'sccl': SCCLModel,
    'umc': UMCModel,
}