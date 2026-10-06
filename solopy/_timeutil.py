from astropy.time import Time


def utc_jd_to_tdb(jd_utc):
    """
    Convert Julian date(s) on the UTC scale to the TDB scale.

    FITS headers written by solopy store ``JD`` as mid-exposure UTC, while kete
    (and therefore skyloc) expects TDB. TDB - UTC is about 69.2 s in 2026.
    """
    return Time(jd_utc, format="jd", scale="utc").tdb.jd
