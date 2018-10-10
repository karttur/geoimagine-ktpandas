'''
Created on 27 apr. 2018

@author: thomasgumbricht
'''


from __future__ import division
import os
import sys

import array as arr
import numpy as np
import geoimagine.support.karttur_dt as mj_dt
import pandas as pd
#import mj_pandas_numba_v73 as mj_pd_numba
from scipy.stats import norm, mstats, stats



def MKtestNumba(x):
    return mj_pd_numba.MKtest(x)

def MKtest(x):  
    n = len(x)
    s = 0
    for k in range(n-1):
        t = x[k+1:]
        u = t - x[k]
        sx = np.sign(u)
        s += sx.sum()
    unique_x = np.unique(x)
    g = len(unique_x)
    if n == g: 
        var_s = (n*(n-1)*(2*n+5))/18
    else:
        tp = np.unique(x, return_counts=True)[1]
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    return z

def TheilSenXY(x,y):  
    res = mstats.theilslopes(x,y)
    return res


def InterpolatePeriodsNumba(ts,dots,steps,filled):
    return mj_pd_numba.InterpolateLinearNumba(ts,dots,steps,filled)

def InterpolateLinearNumba(ts):
    return mj_pd_numba.InterpolateLinearNaNNumba(ts)

def ResamplePeriodsNumba(ts,indexA,resultA):
    return mj_pd_numba.ResampleFixedPeriods(ts,indexA,resultA)

class PandasTS:
    def __init__(self,timestep):
        if timestep in ['monthlyday','monthly','M','MS']:
            self.annualperiods = 12
        elif timestep in ['timespan-MS','timespan-M']:
            self.annualperiods = 12
        elif timestep in ['1D','timespan-1D']:
            self.annualperiods = 365
        elif timestep in ['8D','timespan-8D']:
            self.annualperiods = 46
        elif timestep in ['16D','timespan-16D']:
            self.annualperiods = 23
        elif timestep in ['AS','timespan-A']:
            self.annualperiods = 1
        else:
            print ('timestep',timestep)
            PLEASEADD
        #Initiate a pandas datetimeindex
        
    
    def SetDatesFromPeriod(self,period):
        yyyymmdd = '%(y)d1231' %{'y':period.startdate.year}
        if period.startdate.year == period.enddate.year:
            endfirstyear = period.enddate
        else:
            endfirstyear = mj_dt.yyyymmddDate(yyyymmdd)
        pdTS = pd.date_range(start=period.startdate, end=endfirstyear, freq=period.timestep, closed='left')
        dt = pd.to_datetime(pdTS)
        dt = pdTS.to_pydatetime()
        if dt[dt.shape[0]-1].year > period.startdate.year:
            dt = dt[:-1]
        if period.enddate.year - period.startdate.year > 1:
            for y in range(period.startdate.year+1,period.enddate.year):
                start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':y})
                end = mj_dt.yyyymmddDate('%(y)d0101' %{'y':y+1})
                ts = pd.date_range(start=start, end=end, freq=period.timestep)
                t = ts.to_pydatetime()
                if t[t.shape[0]-1].year > y:
                    t = t[:-1]
                dt = np.append(dt,t, axis=0)
        #and the last year
        start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':period.enddate.year})
        if period.startdate.year < period.enddate.year:
            ts = pd.date_range(start=start, end=period.enddate, freq=period.timestep)
            t = ts.to_pydatetime()
            if t[t.shape[0]-1].year > period.enddate.year:
                t = t[:-1]
            dt = np.append(dt,t, axis=0)
        return dt
    
    def SetMonthsFromPeriod(self,period):
        
        if period.startdate.year == period.enddate.year:
            endfirstyear = period.enddate
            BALLE # Have to check as below moving last date forward
        else:
            yyyymmdd = '%(y)d0131' %{'y':period.startdate.year+1}
            endfirstyear = mj_dt.yyyymmddDate(yyyymmdd)
        pdTS = pd.date_range(start=period.startdate, end=endfirstyear, freq='MS', closed='left')
        dt = pd.to_datetime(pdTS)
        dt = pdTS.to_pydatetime()
 
        if dt[dt.shape[0]-1].year > period.startdate.year:
            dt = dt[:-1]
 
        if period.enddate.year - period.startdate.year > 1:
            for y in range(period.startdate.year+1,period.enddate.year):
                start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':y})
                end = mj_dt.yyyymmddDate('%(y)d0131' %{'y':y+1})
                ts = pd.date_range(start=start, end=end, freq='MS')
                t = ts.to_pydatetime()
                if t[t.shape[0]-1].year > y:
                    t = t[:-1]
                dt = np.append(dt,t, axis=0)
        #and the last year
        start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':period.enddate.year})
        if period.startdate.year < period.enddate.year:
            ts = pd.date_range(start=start, end=period.enddate, freq='MS')
            t = ts.to_pydatetime()
            if t[t.shape[0]-1].year > period.enddate.year:
                t = t[:-1]
            dt = np.append(dt,t, axis=0)
        return dt
    
    def NumpyDate(self,date):
        return pd.to_datetime(np.array([date]))
    
    def SetDatesFromList(self,dateL):
        dt = pd.to_datetime(np.array(dateL))
        self.dateArr = dt.to_pydatetime()
        self.nrYears = int(len(dateL)/self.annualperiods)
        self.yArr = np.ones( ( self.nrYears ), np.float32)
        self.npA = np.arange(self.nrYears)
        self.yArr *= self.npA
        self.olsArr = np.zeros( ( 4 ), np.float32)
        self.yzArr = np.ones( ( [4, self.nrYears] ), np.float32)
        self.yzArr *= self.npA
        
    def SetYYYYDOY(self):
        ydA = []
        refdate = mj_dt.SetYYYY1Jan(2000)
        for d in self.dateArr:
            #print 'refdate',refdate
            #print 'd',d.date()
            deltdays = mj_dt.GetDeltaDays(refdate, d.date())
            ydA.append( deltdays.days )
        self.daydiff20000101 = ydA

        
    def SetModv005DatesFromList(self,dateL):
        print (dateL)
        dateL = [mj_dt.DeltaTime(dt,8) for dt in dateL]
        dt = pd.to_datetime(np.array(dateL))
        self.dateArr = dt.to_pydatetime()
        
    def CreateIndexDF(self,index):
        self.dateframe = pd.DataFrame(index=index)

        #self.df = pd.Series(ts, index=df.index)
        #df1['e'] = Series(np.random.randn(sLength), index=df1.index)

        #print self.df
        #BALLE
        
    def SetDFvalues(self,ts):
        self.df = pd.Series(ts, index=self.dateArr)
        
    def ResampleToAnnualSum(self):
        return self.df.resample('AS').sum()
    
    def ResampleToAnnualSumNumba(self,ts):
        return mj_pd_numba.ToAnnualSum(ts, self.yArr, self.annualperiods, self.nrYears)
    
    def ResampleSumNumba(self,ts,dstArr,dstSize,periods):
        return mj_pd_numba.ResampleToSum(ts, dstArr, dstSize, periods)
    
    def ResampleSeasonalAvgNumba(self,ts):
        #return mj_pd_numba.ToAnnualSum(ts, self.yArr, self.annualperiods, self.nrYears)    
        return mj_pd_numba.ResampleSeasonalAvg(ts, self.yArr, self.annualperiods)
    
    def ExtractMinMaxNumba(self,ts):
        return mj_pd_numba.ExtractMinMax(ts)
    
    def InterpolateLinearSeasonsNaNNumba(self,ts,seasonArr,offset):
        return mj_pd_numba.InterpolateLinearSeasonsNaN(ts,seasonArr,offset,self.annualperiods)
    
    def ResampleToAnnualAverage(self):
        return self.df.resample('AS').mean()
    
    def ResampleToPeriodAverageNumba(self,ts):
        #self.df.resample('AS').mean()
        return mj_pd_numba.ToPeriodAverage(ts, self.yArr, self.annualperiods, self.nrYears)
    
    def ResampleToPeriodStdNumba(self,ts):
        print ('std japp')
        return mj_pd_numba.ToPeriodStd(ts, self.yArr, self.annualperiods, self.nrYears)
    
    def ResampleToPeriodMinNumba(self,ts):
        return mj_pd_numba.ToPeriodMin(ts, self.yArr, self.annualperiods, self.nrYears)
    
    def ResampleToPeriodMaxNumba(self,ts):
        return mj_pd_numba.ToPeriodMax(ts, self.yArr, self.annualperiods, self.nrYears)
    
    def ResampleToPeriodMultiNumba(self,ts):
        #self.df.resample('AS').mean()
        return mj_pd_numba.ToPeriodMulti(ts, self.yzArr, self.annualperiods, self.nrYears)
    
    def AnnualZscoreNumba(self,ts):
        return mj_pd_numba.Zscore(ts, self.yArr)
    
    def AnnualOLSNumba(self,ts):
        return mj_pd_numba.OLSextendedNumba(self.yArr, ts, self.olsArr)
        
if __name__ == "__main__":
    x = np.arange(30)
    x[0] = 1  
    print (x)
    print (MKtest(x))