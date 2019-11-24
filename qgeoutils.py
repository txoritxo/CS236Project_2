import numpy as np

R    = 6371e3
NM2m = 1852
m2NM = 1/NM2m

def distance_bearing_from_position(origin_deg, destination_deg):
# assumed origin as an array where col 0 is latitude and col 1 is longitude in degrees
    lat = 0
    lon = 1
    origin_rad      = origin_deg * np.pi / 180
    destination_rad = destination_deg * np.pi / 180
    delta = destination_rad-origin_rad
    dlat = delta[:,lat]
    dlon = delta[:,lon]
    a = np.sin(dlat/2)**2 + np.cos(origin_rad[:,lat]) * np.cos(destination_rad[:,lat]) * np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c
    brg = np.arctan2(np.sin(dlon)*np.cos(destination_rad[:,lat]),
                   np.cos(origin_rad[:,lat]) * np.sin(destination_rad[:,lat])
                   - np.cos(destination_rad[:,lat]) * np.sin(origin_rad[:,lat]) * np.cos(dlon))
    distance_nautical = d*m2NM
    return distance_nautical, brg

def position_from_distance_bearing(pos_deg, dist_NM, brg_rad):
    c = dist_NM * NM2m / R
    lat1 = pos_deg[:,0] * np.pi / 180
    lon1 = pos_deg[:,1] * np.pi / 180
    lat2 = np.arcsin(np.sin(lat1)*np.cos(c) + np.cos(lat1)*np.sin(c)*np.cos(brg_rad))
    lon2 = lon1 + np.arctan2(np.sin(brg_rad) * np.sin(c) * np.cos(lat1),
                             np.cos(c) - np.sin(lat1) * np.sin(lat2))

    destination = np.hstack((lat2 * 180 / np.pi, lon2 * 180 / np.pi))
    return destination


def test_pos2distance():
    posA = np.array([[40,-4],[41,-5]])
    posB = np.array([[41,-5], [42,-6]])
    d, brg = distance_bearing_from_position(posA, posB)

    posB2 = position_from_distance_bearing(posA, d, brg)
    k=0