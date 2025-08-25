import os
import numpy as np
import pandas as pd
import scipy.signal
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import time

import Tides
import util.coordinate_transforms 

def extract_event_features(events_list, var='x'):
    """
    Processes a list of events to extract pre-slip area and slip severity features for each station.

    Parameters:
        events_list (list of DataFrames): Each event contains time series data from multiple stations.
        derivative (function): A function to compute the derivative (first or second) of station values.
        var (str): Variable passed to `preprocess_events` to select coordinate axis (default 'x').

    Returns:
        new_features (list of DataFrames): One DataFrame per event with features per station.
    """
    # Columns for the features DataFrames
    columns = ["station", "pre-slip_area", "slip_severity", "peak_time", "total_delta", "start_time"]
    
    # Preprocess events, extract displacement
    pre_events = preprocess_events(events_list, var=var)

    # This will hold one feature DataFrame per event
    new_features = []

    # Loop over events
    for i, event in enumerate(pre_events):
        # Initialize an empty DataFrame for this event
        event_features = pd.DataFrame(columns=columns)

        # Get a list of all station columns (excluding the 'time_sec' and start_time column)
        cols = [col for col in event.columns if (col != 'time_sec' and col != 'time_dt')]

        # Loop through each station column
        for col in cols:
            # Compute first and second derivatives of the station's signal
            grad = derivative(event[col].values)
            grad2 = derivative(grad)

            # Identify the peak in the second derivative (maximum acceleration)
            max_idx = np.argmax(np.abs(grad2))
            max_time = event['time_sec'].iloc[max_idx]
            severity = np.abs(grad2[max_idx])

            # compute the pre-slip area (displacement integral up to peak)
            closest_idx = (np.abs(event['time_sec'] - max_time)).idxmin()
            x_segment = event['time_sec'].iloc[:closest_idx + 1].values
            y_segment = event[col].iloc[:closest_idx + 1].values
            integral = np.trapz(y_segment, x_segment)

            # print(f"Adding row for station: {col}, area: {integral}, severity: {severity}")
            
            # Add the extracted features for this station
            event_features.loc[len(event_features)] = {
                "station": col,
                "pre-slip_area": float(integral),
                "slip_severity": severity,
                "peak_time": max_time,
                "total_delta": event[col].iloc[-1],
                "start_time": event['time_dt'].iloc[0]
            }

        # Store the completed features for this event
        new_features.append(event_features)

    return new_features

def derivative(x_col, order=4, crit=.05, spacing=15):
    """
    x_col - col of x values to take derivative of
    order - butterworth filter order
    crit - critical value of butterworth filter
    spacing - spacing of gradient
    """

    y = x_col - np.mean(x_col)

    #1st deriv
    b, a = scipy.signal.butter(order, crit) # butterworth filter 
        
    filtered = scipy.signal.filtfilt(b, a, y, padlen=50) # applies filter, no phase shift
    grad = np.gradient(filtered, spacing) # computes gradient
    return grad


def preprocess_events(raw_events, var='x'):
    """
    Preprocess a list of event DataFrames by aligning and cleaning displacement data.

    Parameters
    ----------
    raw_events : list of pandas.DataFrame
        raw data uploaded into events_list in this case
    var : str, optional
        which axis you want to focus on (x,y,z)

    Returns
    -------
    list of pandas.DataFrame
        A list of cleaned DataFrames where:
        - 'time_sec' gives time in seconds from the start of each event.
        - Only columns ending in `var` and 'time_sec' are retained.
        - Calculate displacement relative to the start of the event
        - Start time of event
        - x cor
        - y cor
    """
    
    processed_events = [] 

    # loop through full raw data
    for event in raw_events:
        # make copy to not worry about editing raw data
        event_clean = event.copy()

        # parse time and calculate seconds from first timestamp (to_datetime and dt.total_seconds)
        event_clean['time_dt'] = pd.to_datetime(event_clean['time'], format='%Y-%m-%d %H:%M:%S')
        event_clean['time_sec'] = (event_clean['time_dt'] - event_clean['time_dt'].iloc[0]).dt.total_seconds()

        # Keep only columns ending with `var` 
        var_cols = [col for col in event_clean.columns if col.endswith(var)]
        event_clean = event_clean[var_cols + ['time_sec']+ ['time_dt']]   # Keep only var cols and time_sec

        # Drop any remaining NaN columns
        event_clean = event_clean.dropna(axis=1)

        # Recalculate displacement relative to first row for each column
        for col in var_cols: # loop over columns
            if col in event_clean.columns: 
                event_clean[col] = abs(event_clean[col] - event_clean[col].iloc[0])

        processed_events.append(event_clean)

    return processed_events

def load_evt(evts_path):
    """
    Load the events into a list of data frames

    Parameters
    ----------
    evts_path: File path to evts files

    Returns
    -------
    List[pandas DataFrame]
        Raw Data
    
    """
    events_list = [] 

    for evt_path in os.listdir(evts_path):
        full_path = os.path.join(evts_path, evt_path)
        # print(f"Loading {evt_path}")
        event = pd.read_csv(full_path, sep="\t")
        
        events_list.append(event)
    return events_list



def plot_event(event: pd.DataFrame, separated=False, var="x") -> None:
    """
    Plot displacement data for an event.
    
    Parameters
    ----------
    event : pd.DataFrame
        Event to plot
    separated : bool
        If True, plot each station in its own subplot
    var : str
        Coordinate axis to plot ('x', 'y', or 'z')
    """
     # Demean and shift to start at 0 
    def demean_to_zero(col):
       
        mean_val = np.mean(col)
        return (col - mean_val) - (col.iloc[0] - mean_val)

    #Return columns ending in the given suffix with any non-NaN values 
    def valid_plot_cols(df, suffix):    
        return [col for col in df.columns if str(col).endswith(suffix) and df[col].notna().any()]

    times = pd.to_datetime(event["time"])
    plot_cols = valid_plot_cols(event, var)
    sta_name_len = 4 # excludes direction

    if not separated:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2_dummy = None

        for plot_col in plot_cols:
            col_data = event[plot_col]
            if not np.isnan(col_data.iloc[0]):
                demeaned = demean_to_zero(col_data)
                ax1.plot(times, demeaned, label=str(plot_col)[:sta_name_len])
                if ax2_dummy is None:
                    ax2_dummy = demeaned

        ax1.set_ylabel(f"{var.upper()} Displacement [m]")
        ax1.set_xlim(times.iloc[0], times.iloc[-1])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax2.plot(times, ax2_dummy)
        ax2.set_xlabel("DateTime")
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 20))
        for label in ax2.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        fig.subplots_adjust(bottom=0.10)
        plt.show()

    else:
        n_cols = 3
        n_rows = math.ceil(len(plot_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), squeeze=False) # always 2d array
        fig.subplots_adjust(hspace=0.5)# spacing

        for i, plot_col in enumerate(plot_cols):
            ax = axes[i // n_cols][i % n_cols] # correct subplot indexing
            demeaned = demean_to_zero(event[plot_col])
            ax.plot(times, demeaned, label=str(plot_col)[:sta_name_len])
            ax.set_title(f"Station {str(plot_col)[:sta_name_len]}")
            ax.set_ylabel(f"{var.upper()} Displacement [m]")
            ax.set_xlim(times.iloc[0], times.iloc[-1])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.legend()

        # Remove unused subplots
        for j in range(len(plot_cols), n_rows * n_cols):
            fig.delaxes(axes[j // n_cols][j % n_cols])

        fig.suptitle(f"{var.upper()} Displacement per Station", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def get_tide_height(days, x_cor, y_cor, start_time):
    """ Get tide height for <days> days from initial date <start_time>, at the coordinates <x_cor> and <y_cor>.

    Parameters
    ----------
    days : int
        Number of days to calculate tide height for.
    x_cor : float
        PS71 x coordinate of tide calculation
    y_cor : _type_
        PS71 y coordinate of tide calculation
    start_time : _type_
        Starting date in %Y-%m-%d %H:%M:%S format

    Returns
    -------
    list[float]
        Tide heights
    """

    ### USER DEFINED PATH TO TIDE MODEL ###
    tide_dir = "/Users/sambrown04/Documents/SURF"
    #######################################
    
    tide_mod = "CATS2008-v2023"
    
    tides = Tides.Tide(tide_mod, tide_dir)
    
    spacing = 1 # every minute

    HR_PER_DAY = 24
    MIN_PER_HR = 60

    dates_timeseries = []
    initial_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    for i in range(days * HR_PER_DAY * MIN_PER_HR // spacing):  # 30 days * 24 hr/day * 60 min/hr * 1/10 calculations/min
        dates_timeseries.append(initial_time + datetime.timedelta(minutes=spacing * i))

    #convert to lon and lat
    lon, lat = util.coordinate_transforms.xy2ll(x_cor, y_cor)
    # print(lon, lat)

    tides = Tides.Tide(tide_mod, tide_dir)
    
    start_time = time.time()
    tide_results = tides.tidal_elevation(
        [lon],
        [lat],
        dates_timeseries,
    ).data.T[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    out = pd.DataFrame(columns = ["time", "tide_height"])
    out.loc[:,"time"] = dates_timeseries
    out.loc[:,"tide_height"] = tide_results

    return out
    

def tide_derivative(tide_df):
    """
    Calculates the derivative of a tide dataset

    Parameters
    ----------
    tide_df: pd.DataFrame
        DataFrame with two columns = ['time', 'tide_height']

    returns: pd.DataFrame
        columns = ['time', 'tide_derivative'] 
        1st derivative will be in units of cm/minute

    NOTE: Time should increment by minutes
    """
    
    # Retrieve our "f" values
    f = tide_df['tide_height'].to_numpy()

    # Paramter 1 signifies there is one minute between tide measurements
    # Since equal distances, this is a standard 2nd-order approx. under the hood.
    deriv = np.gradient(f, 1) 

    # Define our output df
    out = pd.DataFrame(columns = ['time', 'tide_deriv'])
    out['time'] = tide_df['time']

    out['tide_deriv'] = pd.Series(deriv)

    return out


def form_factor_calc(tide_time, days = 3, slide = 1):
    """
    Calculates form factor for tide data

    Parameters
    ----------
    tide_time: pd.DataFrame
        columns = ['time', 'tide_height']

    days: int
        number of days to perform tide calculation on

    slide: int
        calculation parameter

    Returns
    -------
    out: DataFrame
        columns = ['date', 'form_factors']
    """
    reference_time = tide_time['time'].iloc[0]
    seconds = [(date - reference_time).total_seconds() for date in tide_time['time']]
    
    # print(tide_time['tide_height'].shape, tide_time['time'].shape)
    
    tide = tide_time['tide_height']
    dates_timeseries = tide_time['time']
    
    spacing = 4  # Minutes
    mean_days = days
    slide_days = slide
    mean_units = int(mean_days * 24 * 60 / spacing)
    slide_units = int(slide_days * 24 * 60 / spacing)
    
    HR_TO_SEC = 3600
    T_O1 = 25.81933871 * HR_TO_SEC
    T_K1 = 23.93447213 * HR_TO_SEC
    T_M2 = 12.4206012 * HR_TO_SEC
    T_S2 = 12 * HR_TO_SEC
    
    def sines(x, A1, phi1, A2, phi2):
        return A1 * np.sin(2 * np.pi * x / ((T_O1 + T_K1) / 2) + phi1) + A2 * np.sin(
            2 * np.pi * x / ((T_M2 + T_S2) / 2) + phi2
        )
    
    form_factors = []
    dates_form_factor = []
    semidiurnal = []
    diurnal = []
    
    start = 0
    end = mean_units
    while end < len(seconds):
        seconds_tide = np.array(seconds[start:end], dtype=float)
        tide_window = np.array(tide[start:end], dtype=float)
        date_midpoint = dates_timeseries[(start + end) // 2]
        start += slide_units
        end += slide_units
    
        # Fit a sum of sines to the tide
        initial_guess = [50, 0, 50, 0]
        popt, pcov = scipy.optimize.curve_fit(sines, seconds_tide, tide_window, p0=initial_guess)
    
        # Extract fitted parameters
        Diurnal_fit, phi1_fit, SemiDiurnal_fit, phi2_fit = popt
    
        # Generate the fitted curve
        y_fit = sines(seconds_tide, Diurnal_fit, phi1_fit, SemiDiurnal_fit, phi2_fit)
        form_factor = np.abs(Diurnal_fit / SemiDiurnal_fit)
        semidiurnal.append((SemiDiurnal_fit))
        diurnal.append((Diurnal_fit))
    
        form_factors.append(form_factor)
        dates_form_factor.append(date_midpoint)

    # DataFrame to return
    out = pd.DataFrame(columns = ['dates', 'form_factors'])
    out['dates'] = dates_form_factor
    out['form_factors'] = form_factors
    return out

def form_factor_window(tide_time, start_time, duration_minutes = 720):
    """
    Calculates form factor over a single time window (and other features)

    Parameters
    ----------
    tide_time: pd.DataFrame
        columns = ['time', 'tide_height']
    
    start_time: datetime
        Start time of the interval to analyze

    duration_minutes: float
        Duration of the interval in minutes (720 for 12 hours)

    Returns
    -------
    result: list
        [
            form_factor: float,
            diurnal_amplitude: float,
            semidiurnal_amplitude: float,
            diurnal_phase: float,
            semidiurnal_phase: float
        ]
    
    """

    #Constants
    HR_TO_SEC = 3600
    T_O1 = 25.81933871 * HR_TO_SEC
    T_K1 = 23.93447213 * HR_TO_SEC
    T_M2 = 12.4206012 * HR_TO_SEC
    T_S2 = 12 * HR_TO_SEC

    # Define function to fit
    def sines(x, A1, phi1, A2, phi2):
        return A1 * np.sin(2 * np.pi * x / ((T_O1 + T_K1) / 2) + phi1) + A2 * np.sin(
            2 * np.pi * x / ((T_M2 + T_S2) / 2) + phi2
        )

    # Define window we will be using
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)
    window = tide_time[(tide_time['time'] >= start_time) & (tide_time['time'] <= end_time)].copy()

    reference_time = window['time'].iloc[0]

    # Convert each timestamp to a number for fitting
    seconds = np.array([(t - reference_time).total_seconds() for t in window['time']])

    #Extract for faster math operations
    tide_heights = window['tide_height'].to_numpy()

    initial_guess = [50, 0, 50, 0]

    popt, pcov = scipy.optimize.curve_fit(sines, seconds, tide_heights, p0=initial_guess)
    
    A1, phi1, A2, phi2 = popt

    form_factor = abs(A1 / A2)

    return [form_factor, A1, A2, phi1, phi2]


    










