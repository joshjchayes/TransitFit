'''
time_conversions

Module which uses astropy.time to convert between different time standards.
'''

from astropy import time, coordinates as coords, units as u

def MJD_to_BJD(mjd_times, ra, dec, lat, lon, elevation, ra_unit='hourangle'):
    '''
    Takes an array of times in MJD and converts to BJD, using observation and
    telescope data.

    Parameters
    ----------
    mjd_times : array_like, shape (N, )
        The times to be converted from MJD
    ra : tuple, length 3
        The right ascension of the observation in either (d, m, s) or (h, m, s)
        depending on if unit is 'deg' or 'hourangle'
    dec : tuple, length 3
        The declination of the observation in (d, m, s)
    lat : float or str
        The latitude of the observer in degrees or "dd:mm:ss"
    log : float or str
        The longitude of the observer in degrees or "dd:mm:ss"
    elevation : float
        The elevation of the observer in meters
    unit : str, optional
        The unit being used for ra. Either `'deg'` for degrees or `'hourangle'`
        for hour angle. Default is `'hourangle'`.

    Returns
    -------
    bjd_times : array_like, shape (N, )
        The times in Barycentric Julian Date
    '''
    if ra_unit.lower() == 'deg':
        ra_unit = u.deg
    elif ra_unit.lower() == 'hourangle':
        ra_unit = u.hourangle
    else:
        raise ValueError('Unrecognised unit {}'.format(unit))

    ra = coords.Angle(ra, unit=unit)
    dec = coords.Angle(dec, unit=u.deg)

    sky_coord = coords.SkyCoord(ra=ra, dec=dec, frame='icrs')

    obs_location = coords.EarthLocation(lat=lat, lon=lon, height=elevation)

    mjd_times = time.Time(mjd_times, format='mjd', location=obs_location,
                           scale='utc')

    delta_time = mjd_times.light_travel_time(sky_coord)

    bjd_times = mjd_times.tdb + delta_time

    return bjd_times.value
