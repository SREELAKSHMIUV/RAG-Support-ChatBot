import yaml


def load_prompts():

    with open("prompts/prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)

    return prompts