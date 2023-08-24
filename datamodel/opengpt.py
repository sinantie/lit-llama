from pydantic import BaseModel

class Turn(BaseModel):
    user: str
    ai: str    

    def user_to_prompt(self, special_token: str, eos_token: str):
         return f"{special_token} {self.user} {eos_token}"
    
    def ai_to_prompt(self, eos_token: str):
         return f"{self.ai} {eos_token}"
    
    def to_prompt(self, user_token="Human:", ai_token="AI:", eos_token="###"):
        return f"{self.user_to_prompt(user_token, eos_token)} {ai_token}", f"{self.ai_to_prompt(eos_token)}"

    def to_string(self, user_token="Human:", ai_token="AI:", eos_token="###"):
            return " ".join(self.to_prompt(user_token=user_token, ai_token=ai_token, eos_token=eos_token))            
    
class Dialogue(BaseModel):
    turns: list[Turn]

    def to_prompt(self, history_len:int, instruction="", user_token="Human:", ai_token="AI:", eos_token="###"):
        prompt = instruction
        for i in range(history_len - 1):
            prompt += f"{self.turns[i].to_string(user_token, ai_token, eos_token)}\n"        
        last_user, last_ai = self.turns[-1].to_prompt(user_token, ai_token, eos_token)             
        return f"{prompt}{last_user}", last_ai
        
    def to_string(self, instruction=""):
         dialogue = '\n'.join([x.to_string() for x in self.turns])
         return f"{instruction}{dialogue}"