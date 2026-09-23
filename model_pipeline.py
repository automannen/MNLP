# Copied from: https://huggingface.co/docs/transformers/model_doc/nllb

from transformers import AutoModelForSeq2SeqLM


def load_model(checkpoint):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")
    return model


def translate_batch(model, batch, tokenizer, tgt_lang, max_length=128):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    # TODO: what is a good max length
    translated_tokens = model.generate(
        input_ids=input_ids, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        attention_mask = attention_mask,
        max_length=max_length
    )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    

def run_inference(dataloaders, model, tokenizer):
    for direction in dataloaders:
        src_lang, tgt_lang = direction.split('-')
        dataloader = dataloaders[direction]
        for batch in dataloader:
            # debug print statements
            print(batch["references"])
            print(translate_batch(model, batch, tokenizer, tgt_lang))
    # TODO: in which format to return 