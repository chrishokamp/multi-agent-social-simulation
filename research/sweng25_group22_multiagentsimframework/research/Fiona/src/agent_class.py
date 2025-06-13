import ollama
import settings

class Agent:
    def __init__(self, name: str, init_prompt: str):
        self.m_Name = name;
        self.m_Init_Prompt = init_prompt;
        self.m_Rolling_Prompt = init_prompt;
    
    def ask_agent(self, prompt: str, decorated: bool = True) -> ollama.GenerateResponse:
        if (settings.AGENT_REMINDERS_ENABLED):
            prompt = prompt + "\n\n" + self.m_Name + ", it is now your turn to respond!";

        if (settings.AGENT_HOLD_CONTEXT_ENABLED):
            self.m_Rolling_Prompt += prompt;
        else:
            self.m_Rolling_Prompt = prompt;
        
        response = ollama.generate(model=settings.OLLAMA_MODEL, prompt=self.m_Rolling_Prompt);
        if (decorated):
            response['response'] += "\n";

        if (settings.AGENT_HOLD_CONTEXT_ENABLED):
            self.m_Rolling_Prompt += response['response'];
        
        if (decorated):
            response['response'] = self.m_Name + " : " + response['response'];
        return response;
