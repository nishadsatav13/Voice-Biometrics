from tensorflow.keras.models import load_model, Model

# Load the trained model
model = load_model("best_model.h5", compile=False)

# Create embedding model
embedding_model = Model(
    inputs=model.inputs,
    outputs=model.get_layer("embedding_layer").output
)

# Save embedding model
embedding_model.save("embedding_model.h5")

print("✅ embedding_model.h5 created successfully!")
