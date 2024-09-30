# Fake objects
## LLaMA
  - **Tokenizer**
    ```py
    import transformers
    import torch, transformers, tokenizers

    unk_token = '[UNK]'
    image_token = '<image>'

    vocab = {str(id): id for id in range(1, 32064)}
    vocab[unk_token] = 0
    vocab[image_token] = 32000
    tt = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token=unk_token))
    tt.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    print(f"{tt.encode('<image> this is a fake tokenizer').ids = }")
    # tt.encode('<image> this is a fake tokenizer').ids = [32000, 0, 0, 0, 0, 0]

    wrapped_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tt, unk_token=unk_token, clean_up_tokenization_spaces=True, model_input_names=['input_ids', 'attention_mask'])
    wrapped_tokenizer.save_pretrained('test_llama')

    tt = transformers.AutoTokenizer.from_pretrained('test_llama/')
    print(f"{tt('this is a fake tokenizer') = }")
    # tt('this is a fake tokenizer') = {'input_ids': [0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1]}
    ```
  - **Model**
    ```py
    import transformers
    cc = transformers.models.llama.LlamaConfig()
    cc.num_hidden_layers = 2
    mm = transformers.models.llama.LlamaForCausalLM(cc)
    mm.save_pretrained('test_llama')

    mm = transformers.AutoModelForCausalLM.from_pretrained('test_llama/')
    print(f"{mm.generate(inputs=torch.arange(32)[None], max_new_tokens=10).shape = }")
    # mm.generate(inputs=torch.arange(32)[None], max_new_tokens=10).shape = torch.Size([1, 42])
    ```
  - **from_pretrained**
    ```py
    import transformers
    tt = transformers.AutoTokenizer.from_pretrained('test_llama/')
    mm = transformers.AutoModelForCausalLM.from_pretrained('test_llama/', device_map='auto', trust_remote_code=True)
    inputs = tt('hello world', return_tensors='pt')
    print(f"{mm.generate(**inputs, max_new_tokens=5) = }")
    # mm.generate(**inputs, max_new_tokens=5) = tensor([[    0,     0, 29338, 10974, 16694, 27772, 12537]])

    max_new_tokens = 5
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    ret = mm(input_ids=input_ids, attention_mask=attention_mask)
    generated = ret.logits[:, -1:].argmax(axis=-1)
    for _ in range(max_new_tokens - 1):
        ret = mm(input_ids=generated[:, -1:], attention_mask=torch.ones([1, 1]), past_key_values=ret.past_key_values)
        generated = torch.concat([generated, ret.logits[:, -1:].argmax(axis=-1)], dim=-1)
    print(f"{generated = }")
    # generated = tensor([[29338, 10974, 16694, 27772, 12537]])
    ```
## Llava
  - **Tokenizer**
    ```py
    import transformers
    import torch, transformers, tokenizers

    unk_token = '[UNK]'
    image_token = '<image>'
    vocab = {str(id): id for id in range(1, 32064) if id not in [0, 32000]}
    vocab.update({unk_token: 0, image_token: 32000})
    tt = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token=unk_token))
    tt.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    print(f"{tt.encode(f'{image_token} this is a fake tokenizer').ids = }")
    # tt.encode('<image> this is a fake tokenizer').ids = [32000, 0, 0, 0, 0, 0]

    wrapped_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tt, unk_token=unk_token, clean_up_tokenization_spaces=True, model_input_names=['input_ids', 'attention_mask', 'pixel_values'])
    pp = transformers.LlavaProcessor(image_processor=transformers.CLIPImageProcessor(crop_size=336), tokenizer=wrapped_tokenizer)
    pp.save_pretrained('test_llava')

    pp = transformers.AutoProcessor.from_pretrained('test_llava')
    ret = pp(text=f'User: {image_token}\nhello world', images=np.ones([1, 336, 336, 3]), return_tensors='pt')
    print(f"{ret.keys() = }")
    # ret.keys() = dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
    print(f"{ret['input_ids'] = }, {ret['attention_mask'] = }, {ret['pixel_values'].shape = }")
    # ret['input_ids'] = tensor([[    0, 32000,     0,     0]]), ret['attention_mask'] = tensor([[1, 1, 1, 1]]), ret['pixel_values'].shape = torch.Size([1, 3, 336, 336])
    ```
  - **Model**
    ```py
    import transformers
    cc = transformers.models.llava.LlavaConfig()
    cc.text_config.num_hidden_layers = 2
    cc.vision_config.num_hidden_layers = 2
    cc.text_config.vocab_size = cc.vision_config.vocab_size = 32064
    mm = transformers.models.llava.LlavaForConditionalGeneration(cc)
    mm.save_pretrained('test_llava')

    mm = transformers.models.llava.LlavaForConditionalGeneration.from_pretrained('test_llava/')
    input_ids = torch.tensor([1, mm.config.image_token_index, 12728])[None]
    pixel_values = torch.ones([1, 3, 336, 336])
    print(f"{mm.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=5).shape = }")
    # mm.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=5).shape = torch.Size([1, 8])
    ```
  - **from_pretrained**
    ```py
    import transformers

    image_token = '<image>'
    mm = transformers.models.llava.LlavaForConditionalGeneration.from_pretrained('test_llava/')
    pp = transformers.AutoProcessor.from_pretrained('test_llava')
    inputs = pp(text=f'User: {image_token}\nhello', images=np.ones([1, 3, 336, 336]), return_tensors='pt')
    print(f"{mm.generate(**inputs, max_new_tokens=5) = }")
    # mm.generate(**inputs, max_new_tokens=5) = tensor([[1, 0, 32000, 0, 26944, 26944, 12672, 12672, 12672]])
    ```
***
