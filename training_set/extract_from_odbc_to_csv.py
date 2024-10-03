table_name = 'ITINERARY_COMPO'
import pyodbc as od
import pandas as pd
cnxn = od.connect("DSN=% s" % (ec_name))
result = pd.read_sql('select ITINERARY_ID, REVENUE, PAX_INT, OND_AIRPORT_PAIR_ID_STR,TRAFFIC_OPERATOR_ACCOUNT_ID_STR,POINT_OF_ORIGIN_AIRPORT_ID_STR  from %s limit 10000000'%(table_name),cnxn)
len = len(result)
part = int(len/500)
for i in range(500):
    part_result = result.iloc[i*part:(i+1)*part,]
    part_result.to_csv("data_dir/"+ec_name + "_part_" + str(i) + ".csv", index=False, header=False)
