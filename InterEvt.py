# Sam Brown
# sam_brown@mines.edu
# Class for effective standardization of inter-event GPS displacement features.

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

                    
class InterEvt:
    def __init__(self, name, data):
        self.name = name
        self.data = data

        self.avg_disp, self.std_disp = self.get_disp_stats()
        self.avg_r2, self.std_r2, self.avg_slope, self.std_slope = self.get_regression_stats()

    def get_disp_stats(self):
        disps = []
        for event in self.data:
            # X- direction
            col = f"{self.name}_x"
            if col in event and not event[col].isna().any():
                disp = event.iloc[-1][col] - event.iloc[0][col]
                disps.append(disp)
        
        return np.mean(disps), np.std(disps)

    def get_regression_stats(self):
        slopes = []
        r2s = []

        for event in self.data:
            col = f"{self.name}_x"
            if col in event and not event[col].isna().any():
                X = np.array(event.index).reshape(-1, 1)
                y = event[col].values.reshape(-1, 1)

                # Slope and R2 for regression fit on displacement dat between evts
                reg = LinearRegression().fit(X, y)
                slopes.append(reg.coef_[0][0])
                r2s.append(reg.score(X, y))

        return (
            np.mean(r2s), np.std(r2s),
            np.mean(slopes), np.std(slopes)
        )
        
    