import numpy as np
import pandas as pd
import datetime
import csv
import os,sys
import simplekml
import math
import matplotlib.pyplot as plt
import scipy.signal
import qgeoutils as qgeo

max_course = 0
min_course = 0
max_distance = 0
min_distance = 10

def normalize_magnitude(data, min_val, max_val):
    range = abs(max_val-min_val)
    normalized = (data-min_val) / range
    return normalized

def normalize_magnitude2(data, min_val, max_val):
    middle = (max_val+min_val)/2
    range = abs(max_val-min_val)/2
    normalized = (data-middle) / range
    return normalized

# def make_equal_length(data_list):

def normalize_flight_data(data, lon_limits=(-50,-39), lat_limits=(-27,-20), alt_limits=(0,45000)):
    data[:,1] = normalize_magnitude2(data[:,1], lat_limits[0], lat_limits[1])
    data[:,2] = normalize_magnitude2(data[:,2], lon_limits[0], lon_limits[1])
    data[:,3] = normalize_magnitude2(data[:,3], alt_limits[0], alt_limits[1])
    data = np.clip(data,-1.0,1.0)
    return data

def filter_data(data):
    data_next = data[1:,:]
    data_prev = data[0:-1,:]
    diff_data = np.absolute((data_next-data_prev)*60)
    mean_lon = diff_data[:,1].mean()
    mean_lat = diff_data[:, 1].mean()
    thr_lon = 5 * mean_lon
    thr_lat = 5 * mean_lat
    thr = max(thr_lat, thr_lon)
    idx = np.nonzero(diff_data[:,(1,2)] > thr)
    data_next = np.delete(data_next, idx,0)
    return data_next

def code_flight_as_relative_distance_course(data):
# assumed longitude in [1], latitude in [0]
    pnext     = np.flipud(data[1:,])
    pprevious = np.flipud(data[0:-1,])
    d, course = qgeo.distance_bearing_from_position(pprevious, pnext)
    course_diff = np.diff(course)
    course_diff = np.insert(course_diff, 0, 0.0)
    course_diff = np.expand_dims(course_diff, axis=1)
    d = np.expand_dims(d, axis=1)

    data[1:,0] = d[:,0]
    data[1:,1] = course_diff[:,0]

def np_wrap_to_pi(data):
    idx = data <= -np.pi
    data[idx] += np.pi*2
    idx = data > np.pi
    data[idx] -= np.pi*2
    return data

def position_from_relative(data, lon0, lat0, crs0):

    crs = crs0 + np.cumsum(data[:,1])
    dx = data[:,0]*np.sin(crs)
    dy = data[:,0]*np.cos(crs)

def distance_between_points(pa, pb):
    sum_lats_rad = (pa[:,0]+pb[:,0])*math.pi/360
    dy = 12430*np.abs(pa[:,0]-pb[:,0])/180
    dx = 24901*np.abs(pa[:,1]-pb[:,1])/360*np.cos(sum_lats_rad)
    d  = np.sqrt(dx*dx+dy*dy)
    d  = np.expand_dims(d,axis=1)

    course = np.arctan2(dx, dy)
    course_diff = np.diff(course)
    course_diff = np.insert(course_diff, 0, 0.0)
    course = np.expand_dims(np.arctan2(dx, dy), axis=1)
    np_wrap_to_pi(course_diff)
    np_wrap_to_pi(course)

    return d, course_diff, course

def filter_altitude(data):
    min_altitude = data[-1,3]+100
    idx = data[:,3]>min_altitude
    if np.count_nonzero(idx) == 0:
        return None
    data = data[idx]
    return data

def filter_distance_to_destination(data, a_distance, min_elements=None):
    dest = data[-1,[1,2]]
    dest= np.expand_dims(dest,axis=0)
    dist = distance_between_points(data[:,[1,2]], dest)
    idx = np.nonzero(dist[:,0]<a_distance)
    if min_elements is not None:
        num_filtered_elems = idx[0].shape[0]
        if num_filtered_elems < min_elements:
            start_elem = data.shape[0]-min_elements
            idx = np.arange(start_elem, data.shape[0])
            if start_elem < 0:
                return None
    data = data[idx]
    return data

def load_and_filter_data(filename, distance_to_filter, min_elements=None, max_elements=None):
    data = np.load(filename)
    if data.shape[0]==0:
        return None
    data = filter_data(data)
    data = filter_data(data)
    data = filter_data(data)
    data = filter_data(data)
    data = filter_altitude(data)
    if data is None:
        return
    data = filter_distance_to_destination(data, distance_to_filter, min_elements)
    if max_elements is not None and data is not None:
        start_elem = data.shape[0] - max_elements
        data = data[start_elem:,:]
    return data

def flight_family_to_kml(name, file_list):
    kml = simplekml.Kml(open=1)

    for f in file_list:
        data = load_and_filter_data(f,200)
        dirname, filename = os.path.split(f)
        org, dest = split_origin_destination(os.path.split(dirname)[-1])
        linestring = kml.newlinestring(name=org)
        coords = data[:,[2,1]]
        coords = tuple(map(tuple, coords))
        coords = list(coords)
        linestring.coords = coords

    kml.save(name+'.kml')

def test_kml():
    kml = simplekml.Kml(open=1)

    # Create a linestring with two points (ie. a line)
    linestring = kml.newlinestring(name="A Line")
    linestring.coords = [(-122.364383, 37.824664),
                         (-122.365, 37.83),
                         (-122.366, 37.82),
                         (-122.367, 37.83),
                         (-122.368, 37.82)
                         ]

    # Create a linestring that will hover 50m above the ground
    linestring = kml.newlinestring(name="A Hovering Line")
    linestring.coords = [(-122.364167, 37.824787, 50), (-122.363917, 37.824423, 50)]
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground

    # Create a linestring that will hover 100m above the ground that is extended to the ground
    linestring = kml.newlinestring(name="An Extended Line")
    linestring.coords = [(-122.363965, 37.824844, 100), (-122.363747, 37.824501, 100)]
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground
    linestring.extrude = 1

    # Create a linestring that will be extended to the ground but sloped from the ground up to 100m
    linestring = kml.newlinestring(name="A Sloped Line")
    linestring.coords = [(-122.363604, 37.825009, 0), (-122.363331, 37.824604, 100)]
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground
    linestring.extrude = 1

    # Save the KML
    kml.save("test.kml")

def str2datetime(a_date_str):
    return datetime.datetime.strptime(a_date_str, '%Y-%m-%d %H:%M:%S.%f')
    # return datetime.datetime.strptime(a_date_str, '%m/%d/%Y %I:%M:%S %p')

def process_csv_file(fname):
    mat = np.empty([0,5])
    with open(fname) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        time0 = None
        origin = None
        dest   = None
        flight = None

        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
         #ToA(s)	ICAO	Lat(deg)	Lon(deg)	Alt(ft)	EWV(kts)	NSV(kts)	VR(ft/min)	HDG(deg)
            timestr = row['Time']
            current_date = str2datetime(timestr)
            if time0 is None:
                time0 = current_date
                dt = 0
                origin = chr(int(float(row['ORIGIN_CHARACTER_1']))) + chr(int(float(row['ORIGIN_CHARACTER_2']))) + chr(int(float(row['ORIGIN_CHARACTER_3']))) + chr(int(float(row['ORIGIN_CHARACTER_4'])))
                dest =   chr(int(float(row['DESTINATION_CHAR_1'])))+ chr(int(float(row['DESTINATION_CHAR_2'])))+ chr(int(float(row['DESTINATION_CHAR_3'])))+ chr(int(float(row['DESTINATION_CHAR_4'])))
                flight = chr(int(float(row['FLT_NUMBER_CHAR_#1'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#2'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#3'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#4'])))+ \
                         chr(int(float(row['FLT_NUMBER_CHAR_#5'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#6'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#7'])))+ chr(int(float(row['FLT_NUMBER_CHAR_#8'])))

            else:
                dt_ = current_date-time0
                dt = dt_.days*86400+dt_.seconds
            lat  = float(row['PRES_POSN_LAT_FMC/IR'])
            lon  = float(row['PRES_POSN_LONG_FMC/IR'])
            alt  = float(row['ALTITUDE_(1013_25mB)'])
            mach = float(row['MACH'])
            np_row = np.array([float(dt), lat, lon, alt, mach])
            mat = np.vstack((mat, np_row))
            # print('\n{:5d},{:15.6f},{:15.6f},{:8.1f},{:5.2f}'.format(dt, lat, lon, alt, mach))
    return mat, origin, dest, flight

def csv2npy(filein, fileout):
    mat, origin, dest, flight = process_csv_file(filein)
    np.save(fileout, mat)
    print('\nFlight ' + flight + " From: " + origin + " to " + dest)
    print(mat)

def process_and_convert_csv_flight(fname, subDir, rootDir):
    fullFile = os.path.join(subDir,fname)
    mat, origin, dest, flight = process_csv_file(fullFile)
    nullstr = chr(0)+chr(0)+chr(0)+chr(0)
    origin = origin.replace(chr(0), '')
    dest   = dest.replace(chr(0), '')
    origin = 'UNK' if not origin else origin
    dest   = 'UNK' if not dest   else dest
    targetDirName = origin + '_to_' + dest
    fileBareName, _ = os.path.splitext(fname)
    npyFilename = fileBareName + '.npy'
    outFileName = os.path.join(rootDir, targetDirName, npyFilename)
    # print('\nFlight ' + flight + " From: " + origin + " to " + dest)
    os.makedirs(os.path.join(rootDir, targetDirName), exist_ok=True)
    np.save(outFileName, mat)

def batch_process_flight_files(rootDir, startFromFile = None):
    doContinue = startFromFile is None
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if doContinue is True:
                if fname.endswith('.csv'):
                    print('\t Processing ' + fname + ' in Dir ' + dirName)
                    process_and_convert_csv_flight(fname, dirName, rootDir)
                else:
                    print('\t Skipping non csv file ' + fname + ' in Dir ' + dirName)
            else:
                doContinue = startFromFile == fname
                print('\t Skipping file ' + fname + ' in Dir ' + dirName)

def split_origin_destination(s):
    ret = s.split('_to_')
    return ret[0], ret[1]

def count_flights_per_family(rootDir):
    ff=open('nflights.csv','w')
    o=open('norigin.csv','w')
    d=open('ndestination.csv','w')
    dorigin = dict()
    ddestination = dict()

    for dirName, subdirList, fileList in os.walk(rootDir):
        if '_to_' not in dirName:
            continue
        print('Directory: {:s} \t {:d} files'.format(dirName, len(fileList)))
        origin, destination = split_origin_destination(os.path.split(dirName)[-1])
        if origin in dorigin:
            for f in fileList:
                dorigin[origin].append(os.path.join(dirName,f))
        else:
            dorigin[origin] = [os.path.join(dirName,f) for f in fileList]

        if destination in ddestination:
            for f in fileList:
                ddestination[destination].append(os.path.join(dirName,f))
        else:
            ddestination[destination] = [os.path.join(dirName,f) for f in fileList]

        s='{:s},{:s},{:s},{:d}\n'.format(dirName, origin, destination, len(fileList))
        ff.write(s)

    for key,val in dorigin.items():
        s='{:s},{:d}\n'.format(key, len(val))
        o.write(s)

    for key,val in ddestination.items():
        s='{:s},{:d}\n'.format(key, len(val))
        d.write(s)

    ff.close()
    o.close()
    d.close()
    return dorigin, ddestination

def flight_family_to_data(destination):
    all_origin, file_list = count_flights_per_family(
        'C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/2017')
    data_list = []
    l_list = []
    maxlen = 0
    minlen = 1000000
    all_data = np.empty((0,2000,2), dtype='float64')
    for f in file_list[destination]:
        print('filename is ' + f)
        data = load_and_filter_data(f, 200, 2000, 2000)
        if data is None: continue
        data = normalize_flight_data(data)
        data = data[:,[1,2]]
        data = np.expand_dims(data, axis=0)
        all_data = np.append(all_data, data, axis=0)
        # list.append(data)
        l = data.shape[0]
        l_list.append(l)
        maxlen = l if maxlen < l else maxlen
        minlen = l if minlen > l else minlen
        # print(l)

    filename = 'normalized_flights_to_'+destination
    np.save(filename, all_data)

def plot_all_flights_to(destination):
    all_origin, all_destinations = count_flights_per_family('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/2017')
    flight_family_to_kml(destination, all_destinations[destination])

def clip_to_size(data, sz):
    if len(data) < sz:
        return None
    else:
        return(data[-sz:,])

def filter_by_start_position(data, lon_limit=(1.9, 2.55), lat_limit=(49.42,50.23)):
    lat = data[0,0]
    lon = data[0,1]
    if lat_limit[0] <= lat <= lat_limit[1] and lon_limit[0] <= lon <= lon_limit[1]:
        return data
    else:
        return None

def add_linestring(kml, coords, name):
    linestring = kml.newlinestring(name=name)
    coords = tuple(map(tuple, coords))
    coords = list(coords)
    linestring.coords = coords

# def scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
def apply_savgol_filter(data):
    # flt = 1
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.plot(data[:,2], data[:,1], color='Blue')
    lat = np.copy(data[:,1])
    lon = np.copy(data[:,2])
    lat = scipy.signal.savgol_filter(lat, 5, 3,)
    lon = scipy.signal.savgol_filter(lon, 5, 3,)
    data[:,1] = lat
    data[:,2] = lon
    return data
    # ax.plot(data[:, 2], data[:, 1], color='Red')
    # fig.savefig('savgol_lon.png')
    # k=0

def remove_duplicates(data):
    data = data.set_index('time_stamp')
    data = data.loc[~data.index.duplicated(keep='first')]
    return data

def resample_dataframe(data):
    data = data.resample('1s').interpolate(method='polynomial', order=3)
    data = data.dropna()
    return data

def load_adapt_file(filename, samples_per_flight=100):
    lon_limits  = (-2.84, 2.50)
    # lat_limits = (49.45, 52.8)
    lat_limits = (49.45, 51.26)
    data=pd.read_pickle(filename)
    flights_grouped=data.groupby(['part_date_utc','flight'])
    name = 'flights' + str(data['part_date_utc'][0])
    i=0
    kml = simplekml.Kml(open=1)
    # sz = 250
    sz=samples_per_flight
    all_data = np.empty((0,sz,2), dtype='float64')
    flights_processed = 0
    for key, current_flight in flights_grouped:
        i +=1
        # linestring = kml.newlinestring(name=name)
        # coords = current_flight[['longitude', 'latitude']]
        # coords = coords.to_numpy()
        # coords = tuple(map(tuple, coords))
        # coords = list(coords)
        # linestring.coords = coords
        if len(current_flight) < 50: continue
        current_flight = remove_duplicates(current_flight)
        current_flight = current_flight[['latitude', 'longitude', 'altitude']]
        current_flight = resample_dataframe(current_flight)
        time_stamp = current_flight.index.to_numpy()
        resample_dataframe(current_flight)
        time_stamp = time_stamp-time_stamp[0]
        delta_time = time_stamp / np.timedelta64(1, 's')
        # temp_data = current_flight[['latitude', 'longitude', 'altitude']]
        np_data = current_flight.to_numpy()
        np_data = filter_by_start_position(np_data)
        if np_data is None:
            continue
        np_data = apply_savgol_filter(np_data)
        code_flight_as_relative_distance_course(np_data)
        np_data = clip_to_size(np_data, sz)
        if np_data is not None:
            dc_data = np.expand_dims(np_data[:,[1,2]], axis=0)
            all_data  = np.append(all_data, dc_data, axis=0)
            flights_processed +=1
            add_linestring(kml, np_data[:,[2,1]], name)
            continue

        delta_time = np.expand_dims(delta_time, axis=1)
        np_data = np.hstack((delta_time, np_data))
        apply_savgol_filter(np_data)
        norm_data = np.copy(np_data)
        normalize_flight_data(norm_data, lon_limits, lat_limits)
        norm_data = norm_data[:, [1, 2]]
        norm_data = clip_to_size(norm_data, sz)
        if norm_data is not None:
            norm_data = np.expand_dims(norm_data, axis=0)
            all_data  = np.append(all_data, norm_data, axis=0)
            flights_processed +=1
            add_linestring(kml, np_data[:,[2,1]], name)

    kml.save('./kml/'+name+'.kml')
    return all_data, flights_processed

def create_adapt_dataset(rootDir, max_nfiles=1e5, name='GWdataset01', samples_per_flight=100):
    dataset = np.empty((0, samples_per_flight, 2), dtype='float64')  # sequence of length 250 and 2 channels lon & lat
    nfiles_processed = 0
    total_flights = 0
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if fname.endswith('.pickle'):
                print('\nProcessing file ' + fname)
                nfiles_processed +=1
                data , flights_in_file= load_adapt_file(os.path.join(rootDir, fname), samples_per_flight)
                dataset = np.append(dataset, data, axis=0)
                total_flights += flights_in_file
                if nfiles_processed >= max_nfiles: break
            else:
                print('\nSkipping file ' + fname)

    dmax   = np.amax(dataset, axis=0)
    dmin   = np.amax(dataset, axis=0)
    np.save(name,dataset)
    print('\ndataset shape is ' + str(dataset.shape))
    print('\ndataset contains {:d} flights'.format(total_flights))
    # print('\ndistance range is [{:f},{:f}]'.format(dmin[0], dmax[0]))

def split_into_train_test_val(pct_train, pct_val, filename):
    data = np.load(filename)
    numel = len(data)
    idx_train = int(pct_train*numel)
    # idx_

def open_and_plot_npy_flight_file(filename, nflights):
    flights = np.load(filename)
    fig = plt.figure()
    ax = plt.axes()
    for flt in range(nflights):
        ax.plot(flights[flt,:,1], flights[flt,:,0])
        current_trj = flights[flt,]
        current_trj = flights[:,[1,0]]
        name = 'trj_{:d}'.format(flt)
        # np.savetxt(name+'.csv', current_trj, delimiter=',')
    fig.savefig('temp0.png')

def open_and_plot_lon_lat(filename):
    flights = np.load(filename)
    fig = plt.figure()
    ax = plt.axes()
    nflights=1
    for flt in range(nflights):
        ax.plot(flights[flt,:,1])
        ax.plot(flights[flt,:,0])
    fig.savefig('temp_lat_lon.png')

def my_test(ndays):
    init_date = datetime.date(2019,9,1)
    for i in range(ndays):
        newdate=init_date+datetime.timedelta(i)
        print('\n' + str(newdate))


def create_subset_from_file(filename, nflights):
    data = np.load(filename)
    name='subset_{:d}_flights'.format(nflights)
    subset=data[0:nflights,]
    np.save(name, subset)

# batch_process_flight_files('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/04')
# batch_process_flight_files('D:/2017_Flight Data', 'YN111_PR_GEH_GOT_2016_RD0003345971.esb.csv')
# test_csv_file = 'C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/04/YN111_PR_GEH_GOT_2016_RD0003331311.esb.csv'
# test_npy_file = 'C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/04/YN111_PR_GEH_GOT_2016_RD0003331311.esb.npy'
# csv2npy(test_csv_file, test_npy_file)
# count_flights_per_family('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/2017')
# plot_all_flights_to('SBSP')
# flight_family_to_data('SBSP')
# plot_all_flights_to('SBJP')
# test_kml()
# load_and_filter_data('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/2017\SBMG_to_SBSP\YN111_PR_GEH_GOT_2017_RD0003469109.esb.npy',200,2000,2000)
# load_adapt_file('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/Adapt/GW_dataset/flights_2019-04-23.pickle')
# my_test(10)
create_adapt_dataset('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/Adapt/GW_dataset', 1, samples_per_flight=200, name='GW20KF200lDC')
# open_and_plot_npy_flight_file('GW20KF200l.npy', 500)
# open_and_plot_lon_lat('subset_20000_flights.npy')
# create_subset_from_file('GW20KF200l.npy', 20000)