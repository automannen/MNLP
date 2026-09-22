# Copied from: https://huggingface.co/docs/transformers/model_doc/nllb

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", device_map="auto")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt").to(model.device)

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
# Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie