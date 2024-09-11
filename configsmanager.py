import os
import json
import pandas as pd


def check_diff(
    candidate:dict,
    query:dict,
):
    diff = {}
    sum_match_score = 0
    max_match_score = 0
    for k, v in query.items():
        if k not in candidate.keys():
            cv = 'key not found'
            match_score = 0
        else:
            cv = candidate[k]
            match_score = cv == v
        diff[k] = {
            'query': v,
            'candidate': cv,
            'match_score': match_score,
        }
        max_match_score += 1
        sum_match_score += match_score

    return diff, sum_match_score, max_match_score


class ConfigsManager:
    NULL_ID = -1

    def __init__(self, configs_filepath:str):
        self.configs_filepath = configs_filepath
        if os.path.exists(configs_filepath):
            with open(configs_filepath) as h:
                self.configs = json.load(h)
        else:
            self.configs = {}

    
    def next_id(self)->int:
        if len(self.configs.keys()) == 0:
            next_config_id = 1
        else:
            next_config_id = max(self.configs.keys()) + 1
        return next_config_id


    def get_config_id(self, config:dict)->int:
        config_id = ConfigsManager.NULL_ID
        for _config_id, _config in self.configs.items():
            if _config == config:
                config_id = _config_id
                break
        return config_id
    

    def log_config(self, config:dict)->int:
        config_id = self.get_config_id(config=config)
        if config_id == ConfigsManager.NULL_ID:
            next_config_id = self.next_id()
            self.configs[next_config_id] = config
            config_id = next_config_id
        return config_id
    

    def save(self)->bool:
        with open(self.configs_filepath, 'w') as h:
            json.dump(self.configs, h)
        return os.path.exists(self.configs_filepath)
    

    def get_all_config_keys(self):
        all_keys = set()
        for _, config in self.configs:
            all_keys.update(config.keys())
        return list(all_keys)
    

    def get_diff_table(self, query:dict):
        data = { key: [] for key in self.get_all_config_keys() }
        for _id, _config in self.configs.items():
            for k in data.keys():
                if k in ['match (%)', 'config_id']:
                    continue
                if k not in _config.keys():
                    data[k].append(None)
                else:
                    data[k].append(_config[k])
            _, sum_match_score, max_match_score = check_diff(candidate=_config, query=query)
            data['config_id'].append(_id)
            data['match (%)'].append(100 * sum_match_score / max_match_score)
        return pd.DataFrame(data=data).sort_values(by='match (%)', ascending=False).reset_index(drop=True)

