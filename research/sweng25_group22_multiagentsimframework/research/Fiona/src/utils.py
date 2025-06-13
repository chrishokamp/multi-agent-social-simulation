import time
import settings

def current_nano_time():
    return round(time.time() * 1000 * 1000);

def current_milli_time():
    return round(time.time() * 1000);

def time_ns_to_str(time: int):
    if (time < 1000):
        return str(time) + "ns";
    time /= 1000;
    if (time < 1000):
        return str(time) + "ms";
    time /= 1000;
    return str(time) + "s";

def print_response(response):
    if (settings.PRINT_ALL_RESPONSE_DATA_ENABLED):
        print(response);
        print("/n");
    else:
        print(response['response']);

    if (settings.PRINT_RESPONSE_DURATIONS_ENABLED):
        print("*** Load Response Duration = " + time_ns_to_str(response['load_duration']) + " ***");
        print("*** Total Response Duration = " + time_ns_to_str(response['total_duration']) + " ***");