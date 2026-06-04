import tensorflow as tf
print("Bắt đầu load model...")
model = tf.keras.models.load_model("model_best.keras")
print("Load thành công!")
print("Output shape:", model.output_shape)
model.summary()