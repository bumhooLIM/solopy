"""
Bit values of the Lv1 ``MASK`` extension (``MASKVER = 2``).

A pixel is unusable when its value is non-zero; the bits say why. Older Lv1 files
(``MASKVER`` absent) store a plain 0/1 mask without this information.

    >>> from solopy import maskbits
    >>> saturated = (mask & maskbits.SATURATED) != 0
"""

BADPIX = 1        # BPM hot/dead pixel, flat defect, NaN/Inf after calibration, pre-existing mask
SATURATED = 2     # raw value >= SATURATION_ADU
BORDER = 4        # within BORDER_PIX of the frame edge
NONPOSITIVE = 8   # <= 0 after dark subtraction or after flat-fielding
BRIGHT_STAR = 16  # halo of a very bright, round source
TRAIL = 32        # elongated segment (satellite/aircraft trail)

MASKVER = 2
SATURATION_ADU = 3800  # raw ADU; the KL4040 12-bit ADC saturates at 4096
BORDER_PIX = 100

NAMES = {
    BADPIX: "BADPIX",
    SATURATED: "SATURATED",
    BORDER: "BORDER",
    NONPOSITIVE: "NONPOSITIVE",
    BRIGHT_STAR: "BRIGHT_STAR",
    TRAIL: "TRAIL",
}


def header_cards():
    """(keyword, value, comment) cards documenting the bits, e.g. MASKB2 = 'SATURATED'."""
    cards = [("MASKVER", MASKVER, "MASK extension: bit mask (0 = good pixel)")]
    cards += [(f"MASKB{bit}", name, f"MASK bit {bit}") for bit, name in NAMES.items()]
    return cards


def has_bits(header):
    """True if a Lv1 header (primary or MASK) declares a bit mask."""
    return int(header.get("MASKVER", 1)) >= MASKVER
