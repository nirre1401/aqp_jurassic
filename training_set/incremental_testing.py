import datetime
import pandas as pd
data1 = datetime.datetime.now()
data2 = datetime.datetime.now()

diff = data2 - data1

days, seconds = diff.days, diff.seconds
hours = days * 24 + seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
pd.date_range(start='1/1/2018', end='1/08/2018')

a=pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
              dtype='datetime64[ns]', freq='D')

print (hours,minutes,seconds)
import pydbgen
from pydbgen import pydbgen
myDB=pydbgen.pydb()
se=myDB.gen_data_series(data_type='date')
print(se)
testdf=myDB.gen_dataframe(10,['name','city','phone','date'])
base = datetime.datetime.today()
date_list = [base - datetime.timedelta(days=x) for x in range(0, 30)]