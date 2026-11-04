"""Constants, enums, and mappings for OpenSPS."""

from enum import IntEnum

# Total number of parameter gradients in reverse pass
NBGRAD = 33


class BGRAD(IntEnum):
    """Parameter gradient indices (0-indexed) matching BGRAD_* in libopensps."""
    FAGN = 0
    DUST1 = 1
    DUST2 = 2
    DUST3 = 3
    DUST_INDEX = 4
    DUST1_INDEX = 5
    FRAC_OBRUN = 6
    FRAC_NODUST = 7
    MWR = 8
    UVB = 9
    DUSTE_QPAH = 10
    DUSTE_UMIN = 11
    DUSTE_GAMMA = 12
    GAS_LOGZ = 13
    GAS_LOGU = 14
    LOGZSOL = 15
    TAGE = 16
    TAU = 17
    CONST = 18
    FBURST = 19
    TBURST = 20
    SF_START = 21
    ZRED = 22
    SIGMA_SMOOTH = 23
    SF_TRUNC = 24
    SF_SLOPE = 25
    IMF1 = 26
    IMF2 = 27
    IMF3 = 28
    VDMC = 29
    MDAVE = 30
    AGN_TAU = 31
    IGM_FACTOR = 32


# Mapping from parameter string name to BGRAD index
PARAM_TO_BGRAD = {
    "fagn": BGRAD.FAGN,
    "dust1": BGRAD.DUST1,
    "dust2": BGRAD.DUST2,
    "dust3": BGRAD.DUST3,
    "dust_index": BGRAD.DUST_INDEX,
    "dust1_index": BGRAD.DUST1_INDEX,
    "frac_obrun": BGRAD.FRAC_OBRUN,
    "frac_nodust": BGRAD.FRAC_NODUST,
    "mwr": BGRAD.MWR,
    "uvb": BGRAD.UVB,
    "duste_qpah": BGRAD.DUSTE_QPAH,
    "duste_umin": BGRAD.DUSTE_UMIN,
    "duste_gamma": BGRAD.DUSTE_GAMMA,
    "gas_logz": BGRAD.GAS_LOGZ,
    "gas_logu": BGRAD.GAS_LOGU,
    "logzsol": BGRAD.LOGZSOL,
    "tage": BGRAD.TAGE,
    "tau": BGRAD.TAU,
    "const": BGRAD.CONST,
    "fburst": BGRAD.FBURST,
    "tburst": BGRAD.TBURST,
    "sf_start": BGRAD.SF_START,
    "zred": BGRAD.ZRED,
    "sigma_smooth": BGRAD.SIGMA_SMOOTH,
    "sf_trunc": BGRAD.SF_TRUNC,
    "sf_slope": BGRAD.SF_SLOPE,
    "imf1": BGRAD.IMF1,
    "imf2": BGRAD.IMF2,
    "imf3": BGRAD.IMF3,
    "vdmc": BGRAD.VDMC,
    "mdave": BGRAD.MDAVE,
    "agn_tau": BGRAD.AGN_TAU,
    "igm_factor": BGRAD.IGM_FACTOR,
}

BGRAD_TO_PARAM = {v: k for k, v in PARAM_TO_BGRAD.items()}


class OpenSPSError(Exception):
    """Exception raised when libopensps returns a non-zero status code."""

    def __init__(self, status: int, message: str = ""):
        self.status = status
        self.message = message
        super().__init__(f"[libopensps error {status}]: {message}" if message else f"[libopensps error {status}]")
