from tensorflow.keras.models import load_model, Model

# Load trained classifier
model = load_model("best_model.h5", compile=False)

# Show layers (important for debugging)
model.summary()

# Extract from last LSTM layer
embedding_layer = model.layers[3]   # usually last BiLSTM

embedding_model = Model(
    inputs=model.inputs,
    outputs=embedding_layer.output
)

embedding_model.save("embedding_model.h5")

print("✅ New embedding_model.h5 created from LSTM layer!")
