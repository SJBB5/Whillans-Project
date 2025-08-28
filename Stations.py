# sam brown
# sam_brown@mines.edu
# 06/03/2025
# Class for stations to better organize code and possibly allow for more effective standardization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statistics

from Tides import Tide
import util.coordinate_transforms 

import my_lib.funcs

class Station:
    
    def __init__(self, name, evts_path):
        self.name = name
        self.chars = name[0:2]
        self.num = name[2:4]
        self.evts_path = evts_path
        
        self.xcor, self.ycor, self.data = self.preprocess()

        self.pre_slip_a, self.pre_slip_a_sd = self.calc_area()
    
        self.slip_severity, self.slip_severity_sd = self.calc_avg_sv()

        self.slip_size, self.slip_size_sd = self.calc_avg_sz()

        # Commenting this out for runtime reasons for now
        # self.tide_dat = self.get_tide_data()
      
        
    def preprocess(self):
        """
        Retrieve data from only the station

        Returns
        -------
        Average x and y coordinates for the entire time that the station had been transmitting [avg_x, avg_y]
        
        A list of DataFrames with columns for x, y, z, and displacements standardized. 
        If the station is not operational for event, returns empty DF with column 
        'no_event'
        
        """

        
        raw_events = []

        # Conditions for the length of data being processed (one year vs multiple years)
        if isinstance(self.evts_path, str):
            raw_events = my_lib.funcs.load_evt(self.evts_path)
        else:
            for path in self.evts_path:        
                raw_events += my_lib.funcs.load_evt(path)

        processed_events = []

        # Lists for the x and y coordinates
        x_cors = []
        y_cors = []
        
        # loop through full raw data, want only station columns
        for event in raw_events:
            # make copy to not worry about editing raw data
            event_clean = event.copy()
    
            # parse time and calculate seconds from first timestamp (to_datetime and dt.total_seconds)
            event_clean['time_dt'] = pd.to_datetime(event_clean['time'], format='%Y-%m-%d %H:%M:%S')
            event_clean['time_sec'] = (event_clean['time_dt'] - event_clean['time_dt'].iloc[0]).dt.total_seconds()

            # Build expected column names
            x_col = f'{self.name}x'
            y_col = f'{self.name}y'
            z_col = f'{self.name}z'
    
            # Check if all required columns exist and contain data
            if all(col in event_clean.columns for col in [x_col, y_col, z_col]):
                if not (event_clean[x_col].isna().any() or event_clean[y_col].isna().any() or event_clean[z_col].isna().any()):
                    
                    # Store the x and y coordinates
                    x_cors.append(event_clean[x_col].iloc[0])
                    y_cors.append(event_clean[y_col].iloc[0])
                        
                    # Subtract initial value (normalize) and keep columns
                    event_clean[x_col] = abs(event_clean[x_col] - event_clean[x_col].iloc[0])
                    event_clean[y_col] = abs(event_clean[y_col] - event_clean[y_col].iloc[0])
                    event_clean[z_col] = abs(event_clean[z_col] - event_clean[z_col].iloc[0])
    
                    event_clean = event_clean[[x_col, y_col, z_col, 'time_sec', 'time_dt']]
                    processed_events.append(event_clean)
                else:
                    processed_events.append(pd.DataFrame(columns=['no_event']))
            else:
                processed_events.append(pd.DataFrame(columns=['no_event']))

        avg_xcor = sum(x_cors) / len(x_cors)
        avg_ycor = sum(y_cors) / len(y_cors)

        return avg_xcor, avg_ycor, processed_events

        
            
    def calc_area(self, var = 'x'):
        """
        Calculate the pre slip area using trapezoid rule

        Parameters
        ----------
        var: char
            The direction used to calculate pre slip (x is standard)

        Returns
        -------
        average pre-slip: float

        """

        slip_sizes = []
        
        for event in self.data:
            if not 'no_event' in event.columns:
                grad = my_lib.funcs.derivative(event[f'{self.name}{var}'])
                grad2 = my_lib.funcs.derivative(grad)

                # Identify the peak in the second derivative (Sharpest break in signal)
                max_idx = np.argmax(np.abs(grad2))
                max_time = event['time_sec'].iloc[max_idx]
                severity = np.abs(grad2[max_idx])

                # Compute the pre-slip area (displacement integral up to peak)
                closest_idx = (np.abs(event['time_sec'] - max_time)).idxmin()
                x_segment = event['time_sec'].iloc[:closest_idx + 1].values
                y_segment = event[f'{self.name}{var}'].iloc[:closest_idx + 1].values
                integral = np.trapz(y_segment, x_segment)

                slip_sizes.append(integral)
                
        return sum(slip_sizes) / len(slip_sizes), statistics.stdev(slip_sizes)
    
    def calc_avg_sv(self, var = 'x'):
        """
        Calculates the station's average impulsiveness during events using 2nd deriv.

        Parameters
        ----------
        var: char
            The dimension of data used for calculation

        Returns
        -------
        Average impulsiveness: float
        """

        impulsive_list = []
        for event in self.data:
            if not 'no_event' in event.columns:
                
                # Derivatives
                grad = my_lib.funcs.derivative(event[f'{self.name}{var}'])
                grad2 = my_lib.funcs.derivative(grad)

                # Calculation
                max_idx = np.argmax(np.abs(grad2))
                severity = np.abs(grad2[max_idx])

                impulsive_list.append(severity)

        return sum(impulsive_list) / len(impulsive_list), statistics.stdev(impulsive_list)
                
        
        
    def calc_avg_sz(self, var = 'x'):
        displacements = []
        for event in self.data:
            if not 'no_event' in event.columns:
                disp = event.iloc[-1][f'{self.name}{var}']
                displacements.append(disp)

        return sum(displacements) / len(displacements), statistics.stdev(displacements)

        
    def plot_station(self, var='x'):
        var_col = f"{self.name}{var}"
    
        all_x = []
        all_y = []
    
        for event in self.data:
            if 'no_event' not in event.columns and var_col in event.columns:
                x = event['time_sec'].values
                y = event[var_col].values
                plt.plot(x, y, alpha=0.1)
                all_x.append(x)
                all_y.append(y)
    
        if all_x:
            ref_x = all_x[0]
            aligned_ys = [y for x, y in zip(all_x, all_y) if np.array_equal(x, ref_x)]
            if aligned_ys:
                avg_y = np.mean(aligned_ys, axis=0)
                plt.plot(ref_x, avg_y, color='red', linewidth=2, label='Average')
    
        plt.xlabel("Time (sec)")
        plt.ylabel(f"{self.name}{var}")
        plt.title(f"{self.name} - {var.upper()} Axis, All Events and Average")
        plt.legend()
        plt.show()   
            
    #### NOTE: This will only work for stations past or near the grounding line ####
    def get_tide_data(self, var='x', days=30, spacing=10, plot=False):
            
        x_col = f"{self.name}x"
        y_col = f"{self.name}y"
        
        # Loop through preprocessed data to find the first valid event
        for event in self.data:
            if 'no_event' not in event.columns:
                x_cor = event.iloc[0][x_col]
                y_cor = event.iloc[0][y_col]
                start_time = event.iloc[0]['time_dt'].strftime("%Y-%m-%d %H:%M:%S")
                break
    
        # USER DEFINED: Adjust path as needed
        tide_dir = "/Users/sambrown04/Documents/SURF"
        tide_mod = "CATS2008-v2023"
    
        # Generate time series
        initial_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        num_points = days * 24 * 60 // spacing
        dates_timeseries = [initial_time + datetime.timedelta(minutes=spacing * i) for i in range(num_points)]
    
        # Convert to lon/lat
        lon, lat = util.coordinate_transforms.xy2ll(x_cor, y_cor)
    
        # Get tidal elevation
        tides = Tide(tide_mod, tide_dir)
        tide_results = tides.tidal_elevation([lon], [lat], dates_timeseries).data.T[0]
    
        # Optional plot
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dates_timeseries, tide_results, label=f"Station {self.name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Tide Height [cm]")
            plt.legend()
            plt.show()
    
        # Return as DataFrame
        out = pd.DataFrame({
            "time": dates_timeseries,
            "tide_height": tide_results
        })
    
        return out
        


        






