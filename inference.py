from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

text = "India"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

print("Input:", text)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
