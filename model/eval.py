import torch
from peft import PeftModel, PeftConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration

peft_model_name = "/content/lora-flan-t5-sum"
config = PeftConfig.from_pretrained(peft_model_name)

model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_name, device_map={"":0}).cuda()
model.eval()

sample = "(CNN)Nine British citizens were arrested in Turkey on Wednesday, suspected of trying to cross illegally into Syria, the Turkish military said on its website. The group included four children -- the oldest being 10 or 11, with the youngest born in 2013, a Turkish official told CNN on condition of anonymity. The nine were arrested at the Turkey-Syria border, the Turkish military said. It didn't say why the group allegedly was trying to get into Syria, which has been torn by a roughly four-year war between Syrian government forces and Islamist extremist groups and other rebels. Among the war's combatants is ISIS, which has taken over parts of Syria and Iraq for what it claims is its Islamic caliphate, and which is known to have been recruiting Westerners. Accompanying the children were three men and two women; all nine had British passports, the Turkish official said. UK police charge man with terror offenses after Turkey trip . The British Foreign Office said Wednesday that it is aware of reports of the arrests and that it is seeking information about the incident from Turkish authorities. CNN's Gul Tuysuz reported from Istanbul, and Elaine Ly reported from London. CNN's Jason Hanna contributed to this report."
input_ids = tokenizer(sample, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_length=256)
print(f"{sample}")

print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
