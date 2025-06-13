import os
import re
import json
from pathlib import Path

class SimResultsHandler:
    def log_sim_results(results):
        for (i, result) in enumerate(results):
            istr = str(i)
            if (Path('../data/prev_sim_results_' + istr + '.txt').is_file()):
                os.remove('../data/prev_sim_results_' + istr + '.txt')

            if (Path('../data/sim_results_' + istr + '.txt').is_file()):
                os.rename('../data/sim_results_' + istr + '.txt', '../data/prev_sim_results_' + istr + '.txt')

            resultsStr = re.sub(r'(TextMessage\()', r'\n\1', str(result)) 

            with open('../data/sim_results_' + istr + '.txt', 'wt') as f:
                f.write(resultsStr)

    def sim_results_to_json(results, sim_id):
        runs_contents = ""
        num_runs_count = 0
        for result in results:
            num_messages = 0
            chat_log = ""
            output_variables = "{ \"Error\": \"Unassigned\"}"
            for message in result.messages:
                if (message.source == "InformationReturnAgent"):
                    output_variables = str(message.content).replace("\n", "").replace("TERMINATE", "").replace("```json", "").replace("```", "")
                    continue

                num_messages += 1
                if (chat_log != ""):
                    chat_log += ","
                message_content = str(message.content).replace("\n", "").replace("\\\"", "\"").replace("\"", "\\\"")
                chat_log += "{ \"agent\":\"" + str(message.source) + "\", \"message\":\"" + message_content + "\"}"

            is_output_variables_valid_json = output_variables.__contains__("{") and __class__.is_json(output_variables)
            if (num_messages <= 0 or not is_output_variables_valid_json):
                print("ERROR: Run failed, deleting from output")
                continue

            num_runs_count += 1
            if (runs_contents != ""):
                runs_contents += ","
            runs_contents += "{\"num_messages\":" + str(num_messages) + ", \"chat_log\": [" + chat_log + "], \"output_variables\": [" + output_variables + "]}"

        runs = "\"runs\":[" + runs_contents + "]"
        id = "\"id\": \"" + sim_id + "\""
        num_runs = "\"num_runs\":" + str(num_runs_count)
        json_output_str = "{" + id + "," + num_runs + "," + runs + "}"
        #print("\n\n\n\n\n\n" + json_output_str)

        return json.loads(json_output_str)

    def is_json(json_str):
        try:
            json.loads(json_str)
        except ValueError:
            return False
        return True