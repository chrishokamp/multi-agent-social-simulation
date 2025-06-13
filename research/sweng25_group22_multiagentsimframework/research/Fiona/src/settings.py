# This file will be for constants and other settings like which model to use, toggling profiling options, etc

# Debugging
PROFILING_ENABLED = True;
PRINT_ALL_RESPONSE_DATA_ENABLED = False;
PRINT_RESPONSE_DURATIONS_ENABLED = False;

# LLM 
OLLAMA_MODEL = "llama3.2";

# Agents
AGENT_HOLD_CONTEXT_ENABLED = True; # Makes agents remember previous prompts and responses
AGENT_REMINDERS_ENABLED = True; # Reminds agents who they are before they respond
JUDGE_ONE_WORD_RESPONSE_ENABLED = True;
JUDGE_RESPONSE_CSV_PATH = "Test3SimJudgeResponse.csv";


# Simulations
SIMULATION_TURN_LIMIT_ENABLED = True;
SIMULATION_TURN_LIMIT = 2;