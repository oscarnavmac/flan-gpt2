import evaluate

def generate_labels(model, tokenizer, input_list, tagging):
    labels = []
    for input in input_list:
        inputs = tokenizer(input, return_tensors='pt').to(model.get_device())
        input_length = len(tokenizer.decode(inputs["input_ids"][0]))
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=10,
                do_sample=True
            )[0],
            skip_special_tokens=True
        )[input_length:].strip().partition(' ')[0] #.partition(' ')[0] for picking only the first word
        try:
            labels.append(tagging(output))
        except:
            labels.append(-1)

    return labels

def str2bool(str):
    if str.lower() == "true":
        return True
    elif str.lower() == "false":
        return False
    else: raise Exception("Not a Boolean")

def accuracy(references, predictions):
    return evaluate.load("accuracy")