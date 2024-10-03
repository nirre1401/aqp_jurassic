from threading import Thread, Lock
import logging
#import pandasql as pdsql
import multiprocessing
import time
import pandas as pd
from random import randint
import pyodbc as od

result_queue = []
query_counter = 0

dispatcher={'min':pd.DataFrame.min, 'max':pd.DataFrame.max, 'sum': pd.DataFrame.sum, 'avg': pd.DataFrame.mean, 'count': pd.DataFrame.count}
class StoppableThread(Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = Thread.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

class DatabaseWorker(Thread):
    __lock = Lock()

    def __init__(self,query, result, cnn):
        Thread.__init__(self)
        #self.db = db
        self.query = query
        #self.selects = select_cols
        #self.aggs = agg_function
        #self.froms = from_table
        #self.where_dim_cols = where_dim_cols
        #self.dim_op = dim_op
        #self.dim_vals = dim_vals
        self.result = result
        self.connection = cnn
        self.dim_where_indicator = True
        #resultset = self.run()
    def exit(self):
        self.exit()
    def run(self):
        print()
        try:
            self.result = [calc_df_query(query, self.connection)
                           for query
                           in self.query]

            result_queue.append(self.result)
            logging.info("#HUNCH - INFO - Done running batch of %s queries" %(len(self.result)))
            print ("#HUNCH - INFO - Done running batch of %s queries" %(len(self.result)))
        except Exception as e:
            logging.error("#HUNCH - ERROR - failed running a batch of queries %s" % str(e))
            print("#HUNCH - ERROR - failed running a batch of queries %s" % str(e))

def calc_df_query(query, cnn):
    try:
        #if cnn is not None:
        if cnn: # upper condition is for querying the ec itseld (not efficient for many queries)
            #print("querying ec with %s" %(query))
            start = time.clock()
            res = pd.read_sql(query, cnn)
            # if res is None or res.iloc[0,0] is None:
            #     res = 0
            # else:
            #     res = res.iloc[0,0]
            #print('FINISHED RUNNING QUERY IN  ' + str(
             #    round(int(time.clock() - start) / 60, 2)) + ' SECONDS')
            global query_counter
            query_counter = query_counter+1
            if (query_counter%100 ==0):
                print("                                                                                ")
                print("                                                                                ")
                print("                                                                                ")
                print("---------------------------------------------------------------------------------")
                print ("%s Queries had been completed processing" %(query_counter))
                print("________________________________________________________________________________")
                print("                                                                                ")
                print("                                                                                ")
                print("                                                                                ")
                logging.error("#HUNCH - INFO - %s Queries had been completed processing" %(query_counter))


    except Exception as e:
        str_error = ("#HUNCH - ERROR - Unable to run query %s because %s" ) % (str(query), str(e))
        logging.error(str_error)
        print ("#HUNCH - ERROR - Unable to run query %s because %s" ) % (str(query), str(e))
        res = 0
    return (res)
def insert_data_to_globals(value):
    global data
    data = value

def run_multithreaded_queries(queries, data, cnn):
    insert_data_to_globals(data)
    delay = 1
    #result_queue = []
    #queries_results = {}
    workers = {}
    counter = 0

    avail_cpus = multiprocessing.cpu_count()-2
    if avail_cpus > len(queries):
        avail_cpus = len(queries)

    if int(len(queries)/avail_cpus) != 0:
        queries_range = range(0, len(queries) + 1, int(len(queries) / avail_cpus))

    while counter < avail_cpus:
        #key = counter
        key = randint(0, 100000000)

        value = DatabaseWorker(
                                queries[queries_range[counter]:queries_range[counter+1]],
                                None,# result
                                od.connect("DSN=% s" % (cnn)))

        workers[key] = value
        counter = counter+1
    for key, value in workers.items():
        worker = workers.get(key)
        worker.start()
    # for key in range(0, counter, 1):
    #     worker = workers.get(key)
    #     worker.start()
    #
    # Wait for the job to be done
    while len(result_queue) < counter:
        time.sleep(delay)
    job_done = True

    queries_results_list = []
    queries_list = []
    for key, value in workers.items():
        while (value.result is None):
            time.sleep(delay)
        queries_list.extend(value.query)
        queries_results_list.extend(value.result)
        value.connection.close()

    return ([queries_list,queries_results_list])
