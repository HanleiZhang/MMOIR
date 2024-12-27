from .MAG_BERT.manager import MAG_BERT
from .MULT.manager import MULT
from .TCL_MAP.manager import TCL_MAP_manager
from .MMIM.manager import MMIM
from .SDIF.manager import SDIF
from .unsupervised.CC.manager import CCManager
from .unsupervised.MCN.manager import MCNManager
from .unsupervised.UMC.manager import UMCManager
from .unsupervised.USNID.manager import UnsupUSNIDManager
from .unsupervised.SCCL.manager import SCCLManager

method_map = {
    'mag_bert': MAG_BERT,
    'mult': MULT,
    'mmim': MMIM,
    'tcl_map': TCL_MAP_manager,
    'sdif': SDIF,
    'cc': CCManager,
    'mcn': MCNManager,
    'umc': UMCManager,
    'usnid': UnsupUSNIDManager,
    'sccl': SCCLManager
}