import numpy as np
def delaz(eplat,eplong,stlat,stlong, geoco = 0.993277):
    """
    Taken from NMSYN (originally J Tromp's NM code)
    :param eplat: event latitude
    :param eplong: event longitude
    :param stlat: station latitude
    :param stlong: event longitude
    :param geoco: flattening (default: WGS84)
    :return:
    delta (epicentral distance)
    azep (source -> station azimuth counterclockwise from due South)
    azst (station azimuth counterclockwise from due South)
    """
    # Convert geocentric coordinates:
    rad = 2*np.pi/360

    # Latitudes in radians
    el= np.pi/2 - np.arctan(geoco*np.tan(eplat*rad))
    stl=np.pi/2 - np.arctan(geoco*np.tan(stlat*rad))

    # Longitudes in radians
    elon=eplong*rad
    slon=stlong*rad

    AS=np.cos(stl)
    bs=np.sin(stl)
    cs=np.cos(slon)
    ds=np.sin(slon)
    a=np.cos(el)
    b=np.sin(el)
    c=np.cos(elon)
    d=np.sin(elon)

    cdel=a*AS + b*bs*(c*cs+d*ds)

    if np.abs(cdel) > 1:
        if cdel >= 0:
            cdel = 1.0
        else:
            cdel = -1.0
    delt=np.arccos(cdel)
    delta=delt/rad
    sdel=np.sin(delt)
    caze=(AS-a*cdel)/(sdel*b)
    if np.abs(caze) > 1:
        if caze >= 0:
            caze = 1.0
        else:
            caze = -1.0
    aze=np.arccos(caze)
    if bs > 0:
        cazs=(a-AS*cdel)/(bs*sdel)
    if bs == 0:
        if cazs >= 0 :
            cazs = 1
        else:
            cazs = -1
    if np.abs(cazs) > 1:
        if cazs >= 0:
            cazs = 1.0
        else:
            cazs = -1.0
    azs=np.arccos(cazs)
    dif=ds*c-cs*d
    if dif < 0:
        aze = 2*np.pi - aze
    azep=aze/rad
    if dif > 0:
        azs= np.pi*2 - azs
    azst=azs/rad

    azep=180.-azep
    if azep < 0:
        azep= azep + 360.0
    return  delta, azep, azst


def lon_dist_at_lat(lat, rad=6371, geoco=1):
    # Computes distance of 1 degree of longitude at a given latitude:
    colat = np.pi/2 - np.deg2rad(lat)
    if geoco == 1:
        return  2 * np.pi * rad*np.sin(colat)/360
    else:
        raise ValueError()

def dist_lon_at_lat(dist, lat, rad=6371, geoco=1):
    # Computes angular (longitude) distance for a given distance along the surface at given lat
    # Distance should be in km
    colat = np.pi/2 - np.deg2rad(lat)

    if geoco == 1:
        # This is the distance of 1 degree of latitude
        lat1 = 2 * np.pi * rad*np.sin(colat)/360
        return dist/lat1
    else:
        raise ValueError()

