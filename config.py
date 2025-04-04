class PreAttackConfig:
    def __init__(self,
                 model_name = 'gpt-4o',
                 behavior_csv = './data/harmbench.csv',
                 harmful_behav_csv = './data/harmful_behaviors.csv',
                 extract_prompt = './prompts/1_extract.txt',
                 network_prompt = './prompts/2_network.txt',
                 actor_prompt = './prompts/3_actor.txt',
                 query_prompt = './prompts/4_queries.txt',
                 json_format_prompt = './prompts/5_json_format.txt',
                 more_actor_prompt = './prompts/3_more_actor.txt'):
        self.model_name = model_name
        self.behavior_csv = behavior_csv
        self.extract_prompt = extract_prompt
        self.harmful_behav_csv = harmful_behav_csv
        self.network_prompt = network_prompt
        self.query_prompt = query_prompt
        self.actor_prompt = actor_prompt
        self.json_format_prompt = json_format_prompt
        self.more_actor_prompt = more_actor_prompt


class SimulationConfig:
    def __init__(self,
                 model_name = 'gpt-4o',
                 response_state_prompt = './prompts/simulation_unknown.txt',
                 semantic_improvement = './prompts/semantic_improvement.txt',
                 harmful_improvement = './prompts/harmful_improvement.txt',
                 pre_attack_path = './pre_attack'):
        self.model_name = model_name
        self.pre_attack_path = pre_attack_path
        self.response_state_prompt = response_state_prompt
        self.semantic_improvement = semantic_improvement
        self.harmful_improvement = harmful_improvement

        
class InAttackConfig:
    def __init__(self,
                 attack_model_name = 'gpt-4o',
                 target_model_name = 'gpt-4o',
                 modify_prompt = './prompts/attack_modify.txt',
                 pre_attack_path = '',
                 simulation_path = ''):
        self.attack_model_name = attack_model_name
        self.target_model_name = target_model_name
        self.modify_prompt = modify_prompt
        self.pre_attack_path = pre_attack_path
        self.simulation_path = simulation_path
        
class DatasetConfig:
    def __init__(self,
                 attack_data_file = '', 
                 safe_prompt = 'prompts/get_safe_response.txt', 
                 safe_model_name = 'deepseek-chat'):
        self.attack_data_file = attack_data_file
        self.safe_prompt = safe_prompt
        self.safe_model_name = safe_model_name