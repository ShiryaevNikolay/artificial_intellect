import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras


file_name = "dataset.csv"
file_name_with_noise = "dataset_noise.csv"

data_frame = pd.read_csv(file_name)

print(data_frame)

input_data = ["X1", "X2", "X3"]
output_data = ["Y"]

encoders = {
    "X1": lambda value: [float(str(value).replace(',', '.'))],
    "X2": lambda value: [float(str(value).replace(',', '.'))],
    "X3": lambda value: [float(str(value).replace(',', '.'))],
    "Y": lambda value: [float(str(value).replace(',', '.'))],
}


def data_frame_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


def make_supervised(df):
    raw_input_data = df[input_data]
    raw_output_data = df[output_data]
    return {
        "inputs": data_frame_to_dict(raw_input_data),
        "outputs": data_frame_to_dict(raw_output_data)
    }


def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


supervised = make_supervised(data_frame)

encoded_inputs = np.array(encode(supervised["inputs"]))
encoded_outputs = np.array(encode(supervised["outputs"]))

print(encoded_inputs)
print(encoded_outputs)

train_index = 500

train_x = encoded_inputs[:train_index]
train_y = encoded_outputs[:train_index]

test_x = encoded_inputs[train_index:]
test_y = encoded_outputs[train_index:]

model = keras.Sequential([
    keras.layers.Dense(10, input_dim=3, activation="relu"),
    keras.layers.Dense(1),
])

model.compile(
    loss="mse",
    optimizer="adam",
    metrics="accuracy"
)

fit_result = model.fit(
    x=train_x,
    y=train_y,
    batch_size=10,
    epochs=1000,
    validation_split=0.2,
    verbose=0
)

plt.title("Losses train/validation")
plt.plot(fit_result.history["loss"], label="Train")
plt.plot(fit_result.history["val_loss"], label="Validation")
plt.legend()
plt.show()

# plt.title("Accuracies train/validation")
# plt.plot(fit_result.history["accuracy"], label="Train")
# plt.plot(fit_result.history["val_accuracy"], label="Validation")
# plt.legend()
# plt.show()

predicted_test = model.predict(test_x)

real_data = data_frame.iloc[train_index:][input_data + output_data]
real_data["Passembly"] = predicted_test

print(real_data)

model.save_weights("weights.h5")

for layer in model.layers:
    print(layer.get_weights()[0])
