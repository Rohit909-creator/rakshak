from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)

image = Image.open("Japanese_car_accident_blur.jpg")

print("Short Caption:")
print(model.caption(image, length = "short")['caption'])