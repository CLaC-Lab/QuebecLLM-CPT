#!/usr/bin/env python 

import datasets
import os
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import 
from datasets import load_dataset


class Model():
    def __init__(self, model_name, base_model=False):
        self.abs_path = self.get_abs_path(model_name)
        self.peft_config = PeftConfig.from_pretrained(self.abs_path)
        model = AutoModelForCausalLM.from_pretrained(self.peft_config.base_model_name_or_path)
        model = model.to("cuda")s
        self.model = PeftModel.from_pretrained(model,self.abs_path)
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
        
    def generate(self, inputstr):
        a = 1

        

def cole():
    ds = load_dataset("graalul/COLE", "qfrcola", split="test")
    for i in range(10):
        print(ds[i])
    a = 1

if __name__ == "__main__":
    cole()