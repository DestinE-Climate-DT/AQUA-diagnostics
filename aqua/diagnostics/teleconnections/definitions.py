# Parameters in common to the teleconnections are:
#
# telec_type: 'station' or 'region' choose the
#             function to use to compute the teleconnection
# field: field used to compute the teleconnection

NAO_DEFINITIONS = {
    "telec_type": "station",
    "field": "msl",
    "lat1": 37.7,
    "lon1": -25.7,
    "lat2": 64.1,
    "lon2": -22,
}

ENSO_DEFINITIONS = {
    "telec_type": "region",
    "field": "tos",
    "latN": 5,  # ENSO 3.4 region coordinates
    "lonW": -170,
    "latS": -5,
    "lonE": -120,
}

MJO_DEFINITIONS = {  # http://www.bom.gov.au/climate/mjo
    "field": "tnlwrf",
    "latN": 15,
    "latS": -15,
    "lonW": 0,
    "lonE": 360,
    "flip_sign": True,  # To obtain standard OLR variable
}
