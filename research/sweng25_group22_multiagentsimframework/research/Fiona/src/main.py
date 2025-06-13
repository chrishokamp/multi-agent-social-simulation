import settings
import utils
import basic_test
import basic_sim_test
import agent_x3_sim_test
import parallel_test
import parallel_agent_x3_monte_carlo_test
import sys
import info_test
import autogen_test
import os
import pathlib

print("Starting Program. PID=" + str(os.getpid()));
f = open("../data/PID.txt", "w");
f.write("PID=" + str(os.getpid()));
f.close();

ALL_ARGS = "test, sim, 3sim, parallel, parallel3, info, autogen";

if (sys.argv.__len__() == 1):
    print("Error, no argument given for test");
    print("Try arguments: " + ALL_ARGS);
    quit();

numRuns = 1;
if (sys.argv.__len__() == 3):
    numRuns = int(sys.argv[2]);

print("Running " + str(numRuns) + " time(s)");

profiling_average = 0;

for i in range(numRuns):
    # Profiling
    start_time = None;
    end_time = None;
    time_taken = None;
    if (settings.PROFILING_ENABLED):
        start_time = utils.current_nano_time();    

    # Run Tests
    sysArg = sys.argv[1];
    if (sysArg == "test"):
        basic_test.run_test();
    elif (sysArg == "sim"):
        basic_sim_test.run_test();
    elif (sysArg == "3sim"):
        agent_x3_sim_test.run_test();
    elif (sysArg == "parallel"):
        parallel_test.run_test();
    elif (sysArg == "parallel3"):
        parallel_agent_x3_monte_carlo_test.run_test(3);
    elif (sysArg == "info"):
        info_test.run_test();
    elif (sysArg == "autogen"):
        autogen_test.run_test();
    else:
        print("Error, invaid argument");
        print("Try arguments: " + ALL_ARGS);
        quit();

    # Profiling
    if (settings.PROFILING_ENABLED):
        end_time = utils.current_nano_time();
        time_taken = end_time - start_time;
        profiling_average += time_taken;
        print("===================================");
        print("Time taken for response was " + utils.time_ns_to_str(time_taken));
    print("\n");

if (numRuns <= 1):
    quit();

profiling_average /= numRuns;
print("**************************************");
print("Average Time taken for response was " + utils.time_ns_to_str(profiling_average));