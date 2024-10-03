from subprocess import call, Popen
import sys
if __name__ == "__main__":
    num_runs = int(sys.argv[1])
    #p = Popen(["python","training_set_generator.py"])
    #procs = []
    #[procs.append(Popen(["python","training_set_generator.py"])) for i in range(1,num_runs)]
    for i in (range(0,num_runs)):
        print('starting sub process ' + str(i))
        Popen(["python", "training_set_generator_hybrid_GROUPBY.py"]).wait()
        #Popen(["python", "training_set_generator_hybrid_GROUPBY.py"])
        print("---------------------------------------------------------")
        #exit_code = p.wait()
        #print(exit_code)
    #exit_codes = [p.wait() for p in procs]