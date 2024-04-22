from transformers import AutoTokenizer

add_sep = True
pooling_type = "last_token"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer.pad_token = tokenizer.eos_token
if add_sep:
    num_added_tokens = tokenizer.add_special_tokens({"sep_token": "[SEP]"})
if pooling_type == "last_token":
    num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[SUMMARIZE]"})

tokenizer.save_pretrained(".")
