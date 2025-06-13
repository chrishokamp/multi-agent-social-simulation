import ollama
import settings
import agent_class
import utils

def run_test():
    print("Running 3 agent simulation between two agents and a judge");

    shared_init_prompt = "Keep your responses to a single unbroken paragraph. Your response should only involve pure english. ";

    agent_teacher_init_prompt = shared_init_prompt + "You are a teacher, your role is to scold the student for something bad they did. You want to give them detention";
    agent_teacher = agent_class.Agent("Teacher", agent_teacher_init_prompt);

    agent_student_init_prompt = shared_init_prompt + "You are a student, your role is to defend yourself from the teacher who wants to give you detention";
    agent_student = agent_class.Agent("Student", agent_student_init_prompt);

    last_teacher_prompt = ollama.GenerateResponse(response="");
    last_student_prompt = ollama.GenerateResponse(response="");

    rolling_chain = "";

    counter = 0;
    while (True):
        if (settings.SIMULATION_TURN_LIMIT_ENABLED and counter >= settings.SIMULATION_TURN_LIMIT):
            break;
        counter += 1;
        
        last_teacher_prompt = agent_teacher.ask_agent(last_student_prompt['response']);
        utils.print_response(last_teacher_prompt);
        print("\n");

        rolling_chain += last_teacher_prompt['response'];

        if (settings.SIMULATION_TURN_LIMIT_ENABLED and counter >= settings.SIMULATION_TURN_LIMIT):
            break;
        counter += 1;

        last_student_prompt = agent_student.ask_agent(last_teacher_prompt['response']);
        utils.print_response(last_student_prompt);    
        print("\n");

        rolling_chain += last_student_prompt['response'];
    
    print("End of main conversation, asking judge for response\n");

    agent_judge_init_prompt = "You are a judge, you must determine who was the winner of this conversation between a teacher and a student. You must pick exactly one as the victor.";
    if (settings.JUDGE_ONE_WORD_RESPONSE_ENABLED):
        agent_judge_init_prompt += " You must have return a one word response from the following list: [ Teacher, Student ]";
    
    agent_judge = agent_class.Agent("Judge", agent_judge_init_prompt);
    judge_response = agent_judge.ask_agent(rolling_chain, False);
    utils.print_response(judge_response);
    return judge_response;

