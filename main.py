from tensorflow.keras.models import load_model

model = load_model("embedding_model.h5")
print("Embedding model loaded successfully")
print(model.summary())
