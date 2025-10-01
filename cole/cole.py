#!/usr/bin/env python 

import datasets
import huggingface_hub
import os
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftConfig, PeftModel
from datasets import load_dataset

context_str = {
    "task":  "Vous trouverez ci-dessous une instruction décrivant une tâche, éventuellement une entrée fournissant un contexte supplémentaire. Rédigez une réponse courte qui complète la demande.\n",
    "multiple_choice": "Vous trouverez ci-dessous une question à choix multiple. Choissisez le meilleur choix\n"
}

class Model():
    def __init__(self, model_name, base_model=False):
        self.abs_path = self.get_abs_path(model_name)
        print(self.abs_path)
        self.peft_config = PeftConfig.from_pretrained(self.abs_path)
        base_model = AutoModelForCausalLM.from_pretrained(self.peft_config.base_model_name_or_path)
        self.base_model = base_model.to("cuda")
        self.model = PeftModel.from_pretrained(self.base_model,self.abs_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.peft_config.base_model_name_or_path,
            model_max_length=2048,
            padding_side="right",
            trust_remote_code=True,
        )

    def get_abs_path(self, model_name):
        gh_repo = "QuebecLLM-CPT"
        cwd = os.getcwd()
        if cwd.endswith(gh_repo):
            return f"{cwd}/models/{model_name}"
        elif gh_repo in cwd:
            return f"{cwd.split(gh_repo)[0]}/{gh_repo}/models/{model_name}"
        else:
            return model_name    
        
    def generate(self, input_str, context=None):
        #str = "La capitale de la france est"
        str = input_str if context is None else (context_str[context] + input_str)
        inputs = self.tokenizer.encode(
            str, return_tensors = "pt",
        ).to("cuda")
        print(f"Number of input tokens; {inputs.size}")
        text_streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(input_ids = inputs, pad_token_id=self.tokenizer.eos_token_id, streamer = text_streamer, max_new_tokens = 1024,
                    early_stopping=False, repetition_penalty=1.5,
                    use_cache = True, temperature = 0.5, min_p = 0.1)

        
def qfrcore(line):
    str = f"###Question: \"{line["expression"]}\" veut dire\n\n"
    choices = line["choices"]
    #for i in range(len(choices)):
    for i in range(5):
        str += f"({i}) {choices[i]}\n"
    str += "\n###Réponse: \n"
    return str


def cole():
    ds = load_dataset("graalul/COLE","qfrcola", split="test")
    for i in range(10):
        print(ds[i]["label"])
        print(ds[i]["label"])
    #m = Model("3EpochModel")
    #m.generate("Il a beaucoup été question de la personnalité agressive et exigeante de Steve Jobs. [...] Dan’l Lewin, déclare dans ce même magazine que Steve Jobs, durant cette période, « avait des sautes d'humeur inimaginables » [...]\nQuestion: Jobs avait-il des sautes d'humeur inimaginables durant la période où il dirigeait NeXT ??\nRéponse: Oui\n\nAu cours des guerres napoléoniennes, la Royal Navy porta la taille de sa flotte à 175 vaisseaux de ligne et 600 navires au total, ce qui nécessitait 140 000 marins. Alors qu’en temps de paix la Royal Navy était en mesure d’assurer le service de ses navires avec des volontaires, la tâche s’avéra beaucoup plus ardue en temps de guerre. En effet, la Royal Navy entrait alors en concurrence avec des navires marchands et corsaires pour embaucher dans ses équipages le petit groupe de marins expérimentés britanniques, et se servait de la conscription pour faire face au manque de volontaires. Étant donné qu’une grande partie des marins de la marine marchande des États-Unis (estimée à plus de 11 000 en 1805) étaient d’anciens combattants de la marine royale ou des déserteurs, les navires de la Royal Navy décidèrent alors d’intercepter et de fouiller les navires marchands des États-Unis à la recherche de déserteurs. Ceci rendit furieux le gouvernement américain, surtout après l’affaire Chesapeake-Leopard (capture et inspection de l’USS Chesapeake américain par le HMS Leopard britannique).\nQuestion: La Royal Navy avait-elle 175 vaisseaux de ligne pendant les guerres napoléoniennes ??\nRéponse:)")

if __name__ == "__main__":
    cole()