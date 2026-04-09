import xarray as xr

from aqua.core.logger import log_configure


def compute_rho(absso, bigthetao, ref_pressure, loglevel="WARNING"):
    """
    Compute the potential density in-situ.

    Parameters
    ----------
    absso : dask.array.core.Array
        Masked array containing the absolute salinity values (g/kg).
    bigthetao : dask.array.core.Array
        Masked array containing the conservative temperature values (degC).
    ref_pressure : float
        Reference pressure (dbar).

    Returns
    -------
    rho : dask.array.core.Array
        Masked array containing the potential density in-situ values (kg m-3).

    Notes
    -----
    Based on polyTEOS-10. See: https://github.com/fabien-roquet/polyTEOS/blob/36b9aef6cd2755823b5d3a7349cfe64a6823a73e/polyTEOS10.py#L57
    """
    logger = log_configure(loglevel, "compute_rho")
    logger.debug("Computing potential density in-situ.")
    # reduced variables
    SAu = 40.0 * 35.16504 / 35.0  # noqa: N806
    CTu = 40.0  # noqa: N806
    Zu = 1e4  # noqa: N806
    delta_s = 32.0
    ss = xr.ufuncs.sqrt((absso + delta_s) / SAu)
    tt = bigthetao / CTu
    pp = ref_pressure / Zu

    # vertical reference profile of density
    r00 = 4.6494977072e01
    r01 = -5.2099962525e00
    r02 = 2.2601900708e-01
    r03 = 6.4326772569e-02
    r04 = 1.5616995503e-02
    r05 = -1.7243708991e-03
    r0 = (((((r05 * pp + r04) * pp + r03) * pp + r02) * pp + r01) * pp + r00) * pp

    # density anomaly
    r000 = 8.0189615746e02
    r100 = 8.6672408165e02
    r200 = -1.7864682637e03
    r300 = 2.0375295546e03
    r400 = -1.2849161071e03
    r500 = 4.3227585684e02
    r600 = -6.0579916612e01
    r010 = 2.6010145068e01
    r110 = -6.5281885265e01
    r210 = 8.1770425108e01
    r310 = -5.6888046321e01
    r410 = 1.7681814114e01
    r510 = -1.9193502195e00
    r020 = -3.7074170417e01
    r120 = 6.1548258127e01
    r220 = -6.0362551501e01
    r320 = 2.9130021253e01
    r420 = -5.4723692739e00
    r030 = 2.1661789529e01
    r130 = -3.3449108469e01
    r230 = 1.9717078466e01
    r330 = -3.1742946532e00
    r040 = -8.3627885467e00
    r140 = 1.1311538584e01
    r240 = -5.3563304045e00
    r050 = 5.4048723791e-01
    r150 = 4.8169980163e-01
    r060 = -1.9083568888e-01
    r001 = 1.9681925209e01
    r101 = -4.2549998214e01
    r201 = 5.0774768218e01
    r301 = -3.0938076334e01
    r401 = 6.6051753097e00
    r011 = -1.3336301113e01
    r111 = -4.4870114575e00
    r211 = 5.0042598061e00
    r311 = -6.5399043664e-01
    r021 = 6.7080479603e0
    r121 = 3.5063081279e00
    r221 = -1.8795372996e00
    r031 = -2.4649669534e00
    r131 = -5.5077101279e-01
    r041 = 5.5927935970e-01
    r002 = 2.0660924175e00
    r102 = -4.9527603989e00
    r202 = 2.5019633244e00
    r012 = 2.0564311499e00
    r112 = -2.1311365518e-01
    r022 = -1.2419983026e00
    r003 = -2.3342758797e-02
    r103 = -1.8507636718e-02
    r013 = 3.7969820455e-01

    rz3 = r013 * tt + r103 * ss + r003
    rz2 = (r022 * tt + r112 * ss + r012) * tt + (r202 * ss + r102) * ss + r002
    rz1 = (
        (
            ((r041 * tt + r131 * ss + r031) * tt + (r221 * ss + r121) * ss + r021) * tt
            + ((r311 * ss + r211) * ss + r111) * ss
            + r011
        )
        * tt
        + (((r401 * ss + r301) * ss + r201) * ss + r101) * ss
        + r001
    )
    rz0 = (
        (
            (
                (
                    ((r060 * tt + r150 * ss + r050) * tt + (r240 * ss + r140) * ss + r040) * tt
                    + ((r330 * ss + r230) * ss + r130) * ss
                    + r030
                )
                * tt
                + (((r420 * ss + r320) * ss + r220) * ss + r120) * ss
                + r020
            )
            * tt
            + ((((r510 * ss + r410) * ss + r310) * ss + r210) * ss + r110) * ss
            + r010
        )
        * tt
        + (((((r600 * ss + r500) * ss + r400) * ss + r300) * ss + r200) * ss + r100) * ss
        + r000
    )
    r = ((rz3 * pp + rz2) * pp + rz1) * pp + rz0

    # in-situ density
    return r + r0
