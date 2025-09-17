import langdetect
langdetect.DetectorFactory.seed = 0

from transformers import AutoModelForCausalLM, AutoTokenizer
from prefect import task


class Translator:
    def __init__(self):
        model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    @task
    def translate(self, text):
        text = f"""Translate the following :
{text}

Translation :
"""

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            top_p=0.1,
            top_k=5
        )

        outputs = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return outputs

    @task(cache_policy=None)
    def lazy_translate(self, tweets):
        for t in tweets:
            lang = langdetect.detect(t.text)
            if lang != 'en':
                t.text = self.translate(t.text)
