import ollama
import settings
import agent_class
import utils

def run_test():
    print("Running basic simulation between two agents");

    shared_init_prompt = "Keep your responses to a single unbroken paragraph. Your response should only involve pure english. ";

    agent_teacher_init_prompt = shared_init_prompt + "You are a teacher, your role is to scold the student for something bad they did. You want to give them detention";
    agent_teacher = agent_class.Agent("Teacher", agent_teacher_init_prompt);

    agent_student_init_prompt = shared_init_prompt + "You are a student, your role is to defend yourself from the teacher who wants to give you detention";
    agent_student = agent_class.Agent("Student", agent_student_init_prompt);

    last_teacher_prompt = ollama.GenerateResponse(response="");
    last_student_prompt = ollama.GenerateResponse(response="");

    counter = 0;
    while (True):
        if (settings.SIMULATION_TURN_LIMIT_ENABLED and counter >= settings.SIMULATION_TURN_LIMIT):
            break;
        counter += 1;
        
        last_teacher_prompt = agent_teacher.ask_agent(last_student_prompt['response']);
        utils.print_response(last_teacher_prompt);
        print("\n");

        if (settings.SIMULATION_TURN_LIMIT_ENABLED and counter >= settings.SIMULATION_TURN_LIMIT):
            break;
        counter += 1;

        last_student_prompt = agent_student.ask_agent(last_teacher_prompt['response']);
        utils.print_response(last_student_prompt);
        print("\n");

