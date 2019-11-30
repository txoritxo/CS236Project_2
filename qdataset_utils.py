import numpy as np
import pandas as pd
import datetime
import csv
import os,sys
import simplekml
import math
import matplotlib.pyplot as plt
import scipy.signal
import qconfig
import pickle
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

def filter_altitude(data):
    min_altitude = data[-1,3]+100
    idx = data[:,3]>min_altitude
    if np.count_nonzero(idx) == 0:
        return None
    data = data[idx]
    return data

def clip_to_size(data, sz):
    if len(data) < sz:
        return None
    else:
        return(data[-sz:,])

def filter_by_position(data, index=0, lon_limit=(1.9, 2.55), lat_limit=(49.42,50.23)):
    if data is None:
        return None
    lat = data[index,0]
    lon = data[index,1]
    if None in lon_limit or None in lat_limit:
        return data
    if lat_limit[0] <= lat <= lat_limit[1] and lon_limit[0] <= lon <= lon_limit[1]:
        return data
    else:
        return None

def add_linestring(kml, coords, name):
    linestring = kml.newlinestring(name=name)
    coords = tuple(map(tuple, coords))
    coords = list(coords)
    linestring.coords = coords

def apply_savgol_filter(data):
    lat = np.copy(data[:,1])
    lon = np.copy(data[:,2])
    lat = scipy.signal.savgol_filter(lat, 5, 3)
    lon = scipy.signal.savgol_filter(lon, 5, 3)
    data[:,1] = lat
    data[:,2] = lon
    return data

def remove_duplicates(data):
    data = data.set_index('time_stamp')
    data = data.loc[~data.index.duplicated(keep='first')]
    return data

def resample_dataframe(data):
    data = data.resample('1s').interpolate(method='polynomial', order=3)
    data = data.dropna()
    return data

def load_adapt_file(filename, samples_per_flight=100, lon_limits=None, lat_limits=None):
    # lon_limits  = (-2.84, 2.50)
    # # lat_limits = (49.45, 52.8)
    # lat_limits = (49.45, 51.26)
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

def sample_equispaced(samples_per_flight, data):
    numel = len(data)
    idx = np.linspace(0,numel,num=samples_per_flight, dtype=int, endpoint=False)
    new_data = data[idx,]
    return new_data

def remove_slow_speed(data, min_speed):
    return data[data['ground_speed']>=min_speed]

def load_adapt_file2(filename, samples_per_flight=100, nfeatures=2,
                     start_lon_limits=None, start_lat_limits=None,
                     end_lon_limits=None, end_lat_limits=None):
    data=pd.read_pickle(filename)
    flights_grouped=data.groupby(['part_date_utc','flight'])
    name = 'flights' + str(data['part_date_utc'][0])
    i=0
    kml = simplekml.Kml(open=1)
    sz=samples_per_flight
    all_data = np.empty((0,sz,nfeatures), dtype='float64')
    flights_processed = 0
    min_speed=145
    for key, current_flight in flights_grouped:
        i +=1
        if len(current_flight) < samples_per_flight: continue
        current_flight = remove_duplicates(current_flight)
        current_flight = remove_slow_speed(current_flight, min_speed)
        if len(current_flight) < samples_per_flight: continue
        current_flight = current_flight[['latitude', 'longitude', 'altitude']]
        current_flight = resample_dataframe(current_flight)
        time_stamp = current_flight.index.to_numpy()
        time_stamp = time_stamp-time_stamp[0]
        delta_time = time_stamp / np.timedelta64(1, 's')
        # temp_data = current_flight[['latitude', 'longitude', 'altitude']]
        np_data = current_flight.to_numpy()
        np_data = filter_by_position(np_data,index=0, lon_limit=start_lon_limits, lat_limit=start_lat_limits)
        np_data = filter_by_position(np_data, index=-1, lon_limit=end_lon_limits, lat_limit=end_lat_limits)
        if np_data is None:
            continue

        np_data = apply_savgol_filter(np_data)
        np_data = sample_equispaced(samples_per_flight, np_data)
        # np_data = clip_to_size(np_data, sz)
        # norm_data = np.copy(np_data)
        # normalize_flight_data(norm_data, lon_limits, lat_limits)
        # norm_data = norm_data[:, [0, 1]]
        # norm_data = clip_to_size(norm_data, sz)
        if np_data is not None:
            # norm_data = np.expand_dims(norm_data, axis=0)
            aux_data = np.expand_dims(np_data[:,0:nfeatures], axis=0)
            all_data  = np.append(all_data, aux_data, axis=0)
            flights_processed +=1
            add_linestring(kml, np_data[:,[1,0]], name)

    kml.save('./kml/'+name+'.kml')
    # all_data=None
    return all_data, flights_processed

def normalize_dataset(dataset):
    nfeatures = dataset.shape[2]
    temp = np.reshape(dataset, (-1,nfeatures))
    percentilemax = np.percentile(temp, 99, axis=0)+0.05
    percentilemin = np.percentile(temp, 0.5, axis=0)-0.05
    max = np.amax(temp, axis=0)
    min = np.amin(temp, axis=0)
    the_max=percentilemax
    the_min=percentilemin
    # if nfeatures > 2:
    #     the_max = np.array([percentilemax[0], max[1], percentilemax[2]])
    #     the_min = np.array([min[0], percentilemin[1], percentilemin[2]])
    # else:
    #     the_max = np.array([percentilemax[0], max[1]])
    #     the_min = np.array([min[0], percentilemin[1]])
    #
    norm_dataset = normalize_magnitude2(dataset, the_min, the_max)
    gt_than_one = np.transpose(np.where(norm_dataset>1))
    u1, c1 = np.unique(gt_than_one[:,0], return_counts=True)
    gt_than_one_indices = u1[c1>10]
    lt_than_negone = np.transpose(np.where(norm_dataset<-1))
    u2, c2 = np.unique(lt_than_negone[:,0], return_counts=True)
    lt_than_negone_indices = u2[c2>10]
    indices2delete = np.hstack((gt_than_one_indices, lt_than_negone_indices))
    indices2delete = np.unique(indices2delete)
    norm_dataset = np.delete(norm_dataset, indices2delete, axis=0)
    norm_dataset=np.clip(norm_dataset,-1,1)
    return norm_dataset, the_max, the_min

def create_adapt_dataset(rootDir, max_nfiles=1e5, name='GWdataset01', nfeatures = 2, samples_per_flight=100,
                         start_lon_limits=None, start_lat_limits=None,
                         end_lon_limits=None, end_lat_limits=None):
    dataset = np.empty((0, samples_per_flight, nfeatures), dtype='float64')  # sequence of length 250 and 2 channels lon & lat
    nfiles_processed = 0
    total_flights = 0
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if fname.endswith('.pickle'):
                print('\nProcessing file ' + fname)
                nfiles_processed +=1
                data , flights_in_file= load_adapt_file2(os.path.join(rootDir, fname), samples_per_flight,
                                                         start_lon_limits=start_lon_limits, start_lat_limits=start_lat_limits,
                                                         end_lon_limits=end_lon_limits, end_lat_limits=end_lat_limits,
                                                         nfeatures=nfeatures)
                dataset = np.append(dataset, data, axis=0)
                total_flights += flights_in_file
                if nfiles_processed >= max_nfiles: break
            else:
                print('\nSkipping file ' + fname)
    if dataset.size == 0:
        print('\n no dataset was generated. Please check directory of reference data')
        return
    norm_dataset, the_max, the_min = normalize_dataset(dataset)
    data = {'dataset':norm_dataset, 'max':the_max, 'min': the_min}
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.DEFAULT_PROTOCOL)
    np.save(name,norm_dataset)
    print('\ndataset shape is ' + str(dataset.shape))
    print('\ndataset contains {:d} flights'.format(total_flights))

def create_adapt_dataset_from_config(filename):
    print('\nLoading dataset configuration file '+filename)
    cfg = qconfig.parse_dataset_config(filename)
    print('\ndataset configuration parsed')
    create_adapt_dataset(cfg['source'],
                         name=cfg['id'],
                         max_nfiles=cfg['nfiles'],
                         nfeatures=cfg['nfeatures'],
                         samples_per_flight=cfg['nsamples'],
                         start_lon_limits=cfg['start_window']['lon'],
                         start_lat_limits=cfg['start_window']['lat'],
                         end_lon_limits=cfg['end_window']['lon'],
                         end_lat_limits=cfg['end_window']['lat'],
                         )

def split_into_train_test_val(pct_train, pct_val, filename):
    data = np.load(filename)
    numel = len(data)
    idx_train = int(pct_train*numel)
    # idx_

def open_and_plot_npy_flight_file(filename, nflights):
    flights = np.load(filename)
    fig = plt.figure()
    ax = plt.axes()
    max = np.amax(flights)
    min = np.amin(flights)
    print('\n max is {} and min is {}\n'.format(max, min))
    for flt in range(nflights):
        ax.plot(flights[flt,:,1], flights[flt,:,0])
        current_trj = flights[flt,]
        current_trj = flights[:,[1,0]]
        name = 'trj_{:d}'.format(flt)
        # np.savetxt(name+'.csv', current_trj, delimiter=',')
    fig.savefig('temp0.png')

def open_and_plot_pickle_flight_file(filename, nflights):
    pickle_in = open(filename, "rb")
    dict = pickle.load(pickle_in)
    flights = dict['dataset']
    nflights = min(len(flights), nflights)
    plot_n_flights(flights, nflights)

def plot_n_flights(flights, nflights):
    fig = plt.figure()
    ax = plt.axes()
    max = np.amax(flights)
    min = np.amin(flights)
    print('\n max is {} and min is {}\n'.format(max, min))
    for flt in range(nflights):
        ax.plot(flights[flt,:,1], flights[flt,:,0], color='blue', linewidth=0.2)
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

def test_dataset_generation(config_filename,nflights=100):
    create_adapt_dataset_from_config(config_filename)
    open_and_plot_pickle_flight_file('GW20KF200lDC.pickle', nflights)
    # open_and_plot_pickle_flight_file('JFK2LAXtest.pickle', nflights)


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
# create_adapt_dataset('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/Adapt/GW_dataset', 5, samples_per_flight=200, name='GW20KF200lDC', lon_limits  = (-2.84, 2.50),lat_limits = (49.45, 51.26))
# create_adapt_dataset_from_config('./config/dataset_baseline.ini')
# create_adapt_dataset('C:/Users/Carlos/local/development/Stanford/DeepGenerativeModels/project/sw/Data/Adapt/JFK2LAX_dataset', 1, samples_per_flight=10, name='JFK2LAXTest')
# open_and_plot_npy_flight_file('GW20KF200lDC.npy', 100)
# open_and_plot_lon_lat('subset_20000_flights.npy')
# create_subset_from_file('GW20KF200l.npy', 20000)
# test_dataset_generation('./config/dataset_baseline.ini', nflights=100)
# load_adapt_file2('../Data/Adapt/GW_dataset/flights_2018-01-08.pickle', samples_per_flight=200, nfeatures=2)
# open_and_plot_pickle_flight_file('JFK2LAXtest.pickle', 394)

test_dataset_generation('./config/dataset_baseline.ini', nflights=800)

