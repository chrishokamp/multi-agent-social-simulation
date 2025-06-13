import agent_x3_sim_test
import threading
import settings
import pathlib
import csv_grapher

g_mutex = threading.Lock();

def thread_run_sim3():
    result = agent_x3_sim_test.run_test()['response'];
    g_mutex.acquire();
    if (not pathlib.Path("../data/" + settings.JUDGE_RESPONSE_CSV_PATH).is_file()):
        f = open("../data/" + settings.JUDGE_RESPONSE_CSV_PATH, "w");
        f.write("Judge Response\n");
    else:
        f = open("../data/" + settings.JUDGE_RESPONSE_CSV_PATH, "a");        
    f.write(result + "\n");
    f.close();
    g_mutex.release();


def run_test(numTests: int):
    threads = [];

    for i in range(numTests):
        threads.append(threading.Thread(target=thread_run_sim3));

    print("Starting threads");
    for i in range(numTests):
        threads[i].start();

    print("Joining threads");
    for i in range(numTests):
        threads[i].join();

    print("All threads joined");

    csv_grapher.graph_csv_1col_freq("../data/" + settings.JUDGE_RESPONSE_CSV_PATH);
