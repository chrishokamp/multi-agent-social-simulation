import ollama
import settings
import utils
import threading

def generate(prompt: str):
    response = ollama.generate(model=settings.OLLAMA_MODEL, prompt=prompt);
    utils.print_response(response); 

TEST_TOPICS = ["Fish", "Tigers", "Cows", "Alligators", "Books", "Movies", "Poles"];
NUM_THREADS = 8;
USE_DIFFERENT_TOPICS = False;

# 1 thread took 4s
# 2 threads took 6.5s (-1.5 from serial, 120% faster)
# 3 threads took 8.5s (-3.5s from serial, 140% faster)
# 4 threads took 12s (-4s from serial, 130% faster)
# 5 threads took 16s (-4s from serial, 125% faster)
# 6 threads took 18s (-6s from serial, 130% faster)
# 7 threads took 21s (-7s from serial, 130% faster)
# 8 threads took 23s (-9s from serial, 140% faster)

# On Desktop, 8 threads took 5.6s 

def run_test():
    print("Running multiple generations in parallel");

    threads = [];

    for i in range(NUM_THREADS):
        topic = TEST_TOPICS[i] if USE_DIFFERENT_TOPICS else TEST_TOPICS[0];
        threads.append(threading.Thread(target=generate, args=("List 10 " + topic,)));

    print("Starting threads");
    for i in range(NUM_THREADS):
        threads[i].start();

    print("Joining threads");
    for i in range(NUM_THREADS):
        threads[i].join();

    print("All threads joined");

    

