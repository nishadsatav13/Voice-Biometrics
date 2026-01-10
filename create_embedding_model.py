from tensorflow.keras.models import load_model, Model

# 1. Load the trained model
model = load_model("best_model.h5", compile=False)

# 2. Build embedding-only model directly from layers
embedding_layer = model.get_layer("embedding_layer")

embedding_model = Model(
    inputs=model.inputs,          # <-- use model.inputs (not model.input)
    outputs=embedding_layer.output
)

# 3. Save embedding model
embedding_model.save("embedding_model.h5")
print("✅ embedding_model.h5 created successfully!")
