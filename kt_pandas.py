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
        if not timestep:
            self.annualperiods = 0 
        elif timestep == 'static':
            self.annualperiods = 0
        elif timestep in ['monthlyday','monthly','M','MS']:
            self.annualperiods = 12
        elif timestep in ['timespan-MS','timespan-M']:
            self.annualperiods = 12
        elif timestep in ['D','timespan-D']:
            self.annualperiods = 365
        elif timestep in ['8D','timespan-8D','seasonal-8D']:
            self.annualperiods = 46
        elif timestep in ['16D','timespan-16D','seasonal-16D']:
            self.annualperiods = 23
        elif timestep in ['1D']:
            self.annualperiods = 365
        elif timestep in ['A','AS','timespan-A','singleyear']:
            self.annualperiods = 1
        elif timestep in ['staticmonthly','seasonal-M','seasonal-Mday','static-MS','static-M']:
            self.annualperiods = 12
        elif timestep in ['autocorr-MS','autocorr-M']:
            self.annualperiods = 12
        elif timestep in ['allscenes','anyscene']:
            self.annualperiods = -1
        else:
            print ('timestep',timestep)
            PLEASEADD
        
    def SetDatesFromPeriod(self,period):
        yyyymmdd = '%(y)d1231' %{'y':period.startdate.year}
        if period.startdate.year == period.enddate.year:
            endfirstyear = period.enddate
        else:
            endfirstyear = mj_dt.yyyymmddDate(yyyymmdd)
            #add one day, so the stepping ends at the last day, not one day before
        endfirstyear = mj_dt.DeltaTime(endfirstyear, 1)

        pdTS = pd.date_range(start=period.startdate, end=endfirstyear, freq=period.timestep, closed='left')

        #dt = pd.to_datetime(pdTS)
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
        self.pdDates = pd.DatetimeIndex(data=dt)
        '''
        print (pdTS)
        print (self.pdDates)
        print (self.pdDates.shape)
        print ('dt',dt)
        print ('dt',dt.shape)
        BALLE
        '''
        return dt
    
    def SetDatesFromPeriodEnds(self,period,step):
        '''
        '''
        startdate = mj_dt.DeltaTime(period.startdate, step-1)
        enddate = mj_dt.DeltaTime(period.enddate, step-1)

        yyyymmdd = '%(y)d1231' %{'y':startdate.year}
        endfirstyear = mj_dt.yyyymmddDate(yyyymmdd)
        endfirstyear = mj_dt.DeltaTime(endfirstyear, step-1)
        #yyyymmdd = mj_dt.DeltaTime(yyyymmdd, step-1)
        
        if period.startdate.year == enddate.year:
            endfirstyear = enddate
        #else:
        #    endfirstyear = mj_dt.yyyymmddDate(yyyymmdd)
        pdTS = pd.date_range(start=startdate, end=endfirstyear, freq=period.timestep, closed='left')
        #dt = pd.to_datetime(pdTS)
        dt = pdTS.to_pydatetime()
        if dt[dt.shape[0]-1].year > startdate.year:
            dt = dt[:-1]
        if enddate.year - startdate.year > 1:
            for y in range(startdate.year+1,enddate.year):
                start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':y})
                end = mj_dt.yyyymmddDate('%(y)d0101' %{'y':y+1})
                ts = pd.date_range(start=start, end=end, freq=period.timestep)
                t = ts.to_pydatetime()
                if t[t.shape[0]-1].year > y:
                    t = t[:-1]
                #print ('appending inside year',t)
                dt = np.append(dt,t, axis=0)
        #and the last year
        start = mj_dt.yyyymmddDate('%(y)d0101' %{'y':enddate.year})
        #start = mj_dt.DeltaTime(start, step-1)
        #print ('start, enddate',start, enddate)
        if startdate.year < enddate.year:
            if enddate > start:
                ts = pd.date_range(start=start, end=enddate, freq=period.timestep)
                t = ts.to_pydatetime()
                if t[t.shape[0]-1].year > enddate.year:
                    t = t[:-1]
                dt = np.append(dt,t, axis=0)
                #print ('appending last year',t)
        return dt
    
    def SetMonthsFromPeriod(self,period):
        
        if period.startdate.year == period.enddate.year:
            endfirstyear = period.enddate
            #print (endfirstyear)
            #BALLE # Have to check as below moving last date forward
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
        self.pdDates = pd.DatetimeIndex(data=dt)
        return dt
    
    def SetYearsFromPeriod(self,period):
        dt = []
        if period.enddate.month == 12:
            for y in range (period.startdate.year,period.enddate.year+1):
                dt.append(y)
        else:
            for y in range (period.startdate.year,period.enddate.year):
                dt.append(y)

        return dt
 
    def _SetYear(self):
        #print self.pdDates
        #print self.annualstep
        self.year_start_dates = self.pdDates[self.pdDates.is_year_start]
        self.yearD = {}
        self.yearL = []
        #print self.year_start_dates
        for y in self.year_start_dates:
            #print 'and',self.df[self.df.date == year].index[0]
            
            #print ('y',y)
            year = y.year
            self.yearL.append(year)
            '''
            i = self.df[self.df.date == y].index[0]
            self.yearD[i] = year
            '''
            #print 'year', y,self.df[self.df.date == year].index[0],self.df['datestr'][i]
        #BALLE
        #create a dict that holds the number in the ts, and the year
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