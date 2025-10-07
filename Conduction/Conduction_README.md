<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/e2978b64-444c-44ba-ab6b-00af1e909931" />
<img width="1035" height="882" alt="image" src="https://github.com/user-attachments/assets/14de4632-cd55-42dd-a2b4-1f222125d0ed" />
 Dataset

The dataset you will require: [thermal_data_100.csv](thermal_data_100.csv)

The steps you need to follow:
Just go line by line of the code to understand the basics of machine learning.


# Section 1: Import Required Libraries

*We will start by importing the Python libraries that help us build and train machine learning models, handle data, and create plots for visualization.*

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

*In this step, we will upload a dataset into our Colab environment and load it into Python. After that, we’ll take a quick look at the data to understand what it contains.*


    uploaded = files.upload()  # Upload 'thermal_data_100.csv'
    
    data = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

# Section 3: Define Inputs and Outputs

*Now, we will identify the input features (the data the model will use to make predictions) and the output labels (what we want the model to predict). This is an important step for setting up a supervised machine learning task.*



    input_cols = ["T0", "T1", "T2", "T3", "T4", "T5"]
    output_cols = ["q", "T1*", "T2*", "T3*", "T4*"]

    
# Section 4: Split Data into Training and Test Sets

*Next, we will divide our dataset into training and testing sets. The training set is used to teach the model, while the testing set is used to see how well the model performs on new, unseen data. This helps us evaluate the model’s accuracy and reliability.*

    train_dataset = data.sample(frac=0.8, random_state=42)
    test_dataset = data.drop(train_dataset.index)

# Section 5: Normalize Data

*Neural networks work better when all input features are on a similar scale. Normalization rescales the data so that values are neither too big nor too small. This helps the model learn faster, converge more reliably, and avoid giving more importance to features just because they have larger numbers.*


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

*Now we will create our neural network. A neural network is made up of layers of neurons that process the input data. Each layer uses an activation function to help the network learn complex patterns. We’ll define the input layer (which takes our features), any hidden layers (which process the information), and the output layer (which makes predictions).*


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

*Before training the model, we define training parameters like the number of epochs (how many times the model will see the full dataset) and callbacks (tools that help monitor or control training, like stopping early if the model stops improving). These settings help the model train efficiently and avoid overfitting.*

    Callback to print progress
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0:
                    print('')
                print('.', end='')
        
        EPOCHS = 1000

# Section 8: Example Prediction Before Training

*Before we train the model, we can make a test prediction using random, untrained weights. This shows that initially, the model’s predictions are basically random and not accurate. It helps you understand how much the model improves after training.*


    example_batch = normed_train_data[:5].values
    example_result = baseline_model.predict(example_batch)
    print("Example predictions (q, T1*, T2*, T3*, T4*):")
    print(example_result)

# Section 9: Setup TensorBoard

*TensorBoard is a tool that lets us visualize and monitor our model’s training in real time. We can track metrics like loss and accuracy, see how the model improves over epochs, and spot problems early. Setting it up helps us understand what’s happening inside the neural network as it learns.*


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Section 10: Train the Model

*Now we will train (fit) the model on our data. During training, the model learns the patterns in the input features to make better predictions. We can also use a validation split to check the model’s performance on unseen data while training, and callbacks to monitor progress or stop training early if needed. This step is where the model actually “learns.”*

    
    history = baseline_model.fit(
        normed_train_data.values, train_labels.values,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback, PrintDot()]
    )

# Section 11: Plot Training History

*After training, we can visualize the model’s learning curves using metrics like MAE (Mean Absolute Error) and MSE (Mean Squared Error). These plots show how the model’s predictions improve over time and help us understand if the model is learning well or if it might be overfitting.*

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

*Finally, we will use the model to make predictions on the test data and compare them to the actual values. We can create a scatter plot to visualize how close the predictions are to the true values. This helps us see the model’s performance in a clear, visual way.*


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

*We can also look at histograms of the prediction errors (the differences between predicted and actual values). This helps us see where the model is making bigger mistakes and understand its strengths and weaknesses for each output.*

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
