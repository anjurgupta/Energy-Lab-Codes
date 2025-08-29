<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/e2978b64-444c-44ba-ab6b-00af1e909931" />
<img width="1035" height="882" alt="image" src="https://github.com/user-attachments/assets/14de4632-cd55-42dd-a2b4-1f222125d0ed" />
 Dataset

The dataset you will require: [thermal_data_100.csv](thermal_data_100.csv)

The steps you need to follow:
Just go line by line of the code to understand the basics of machine learning.


# Section 1: Import Required Libraries

(Libraries needed for ML, data handling, and plotting.)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    Import preprocessing layers individually

    from tensorflow.keras.layers import Normalization, Rescaling, TextVectorization

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import io
    import datetime
    from google.colab import files

Make numpy printouts easier to read
    np.set_printoptions(precision=3, suppress=True)

# Section 2: Upload and Load Dataset

(Bring external data into Python and inspect it.)


    uploaded = files.upload()  # Upload 'thermal_data_100.csv'
    
    data = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

# Section 3: Define Inputs and Outputs

(Input features and output labels for supervised ML.)

Define input and output columns

    input_cols = ["T0", "T1", "T2", "T3", "T4", "T5"]
    output_cols = ["q", "T1*", "T2*", "T3*", "T4*"]

    
# Section 4: Split Data into Training and Test Sets

(Train/test split to evaluate model performance.)

    train_dataset = data.sample(frac=0.8, random_state=42)
    test_dataset = data.drop(train_dataset.index)

# Section 5: Normalize Data

(why normalization is important for neural networks?)


    Training stats (for normalization)
        train_stats = train_dataset[input_cols].describe().transpose()
    
    Normalize function
        def normalize(df):
            return (df - train_stats['mean']) / train_stats['std']
    
    normed_train_data = normalize(train_dataset[input_cols])
    normed_test_data = normalize(test_dataset[input_cols])

    Separate labels (outputs)
        train_labels = train_dataset[output_cols]
        test_labels = test_dataset[output_cols]
    
    Display normalized data
        print("Normalized train data (first 5 rows):")
        print(normed_train_data.head())
        print("\nNormalized test data (first 5 rows):")
        print(normed_test_data.head())


# Section 6: Build the Model

(neural network architecture: layers, activation functions, inputs, and outputs.)


    num_outputs = 5
    
    baseline_model = keras.Sequential([
        layers.Input(shape=(len(input_cols),)),  # Use Input layer instead of input_shape in Dense
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_outputs)  # 5 outputs
    ])

    Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        baseline_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        baseline_model.summary()


# Section 7: Define Callback and Training Parameters



    Callback to print progress
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0:
                    print('')
                print('.', end='')
        
        EPOCHS = 1000

# Section 8: Example Prediction Before Training

(how model predictions work before any training.)



    example_batch = normed_train_data[:5].values
    example_result = baseline_model.predict(example_batch)
    print("Example predictions (q, T1*, T2*, T3*, T4*):")
    print(example_result)

# Section 9: Setup TensorBoard

(Monitoring training metrics interactively.)


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Section 10: Train the Model

(Model fitting, validation split, and callbacks.)

    
    history = baseline_model.fit(
        normed_train_data.values, train_labels.values,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback, PrintDot()]
    )

# Section 11: Plot Training History

( visualize learning curves (MAE, MSE) to understand model performance.)

    def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # MAE
    plt.figure()
    plt.plot(hist['epoch'], hist['mae'], label='Train MAE')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

    # MSE
    plt.figure()
    plt.plot(hist['epoch'], hist['mse'], label='Train MSE')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()

    plot_history(history)


# Section 12: Test Predictions and Scatter Plot

(Visualize model predictions vs actual values.)


    baseline_test_predictions = baseline_model.predict(normed_test_data.values)
    
    plt.figure()
    plt.scatter(test_labels['q'], baseline_test_predictions[:, 0])
    plt.xlabel("Test Values [q]")
    plt.ylabel("Predictions [q]")
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100], 'r')
    plt.show()

# Section 13: Prediction Error Histograms

(check model errors for each output.)


    baseline_error = baseline_test_predictions - test_labels.values
    output_names = output_cols

    plt.figure(figsize=(15, 10))
    for i, name in enumerate(output_names):
        plt.subplot(3, 2, i+1)
        plt.hist(baseline_error[:, i], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel(f"Prediction Error [{name}]")
        plt.ylabel("Count")
        plt.title(f"Error Distribution for {name}")
    
    plt.tight_layout()
    plt.show()
