from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "k", "v"],
 lora_dropout=0.1,
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

output_dir="lora-flan-t5-sum"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    num_train_epochs=1,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="tensorboard"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
    eval_dataset=valid_tokenized_dataset
)
model.config.use_cache = False

trainer.train()
trainer.evaluate(eval_dataset=valid_tokenized_dataset)

peft_save_model_name="lora-flan-t5-sum"
trainer.model.save_pretrained(peft_save_model_name)
tokenizer.save_pretrained(peft_save_model_name)
trainer.model.base_model.save_pretrained(peft_save_model_name)
