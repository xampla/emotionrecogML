import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

#FILES
#TRAINING_IMAGES = "./data/fer2013_images/Training"
#PUBLIC_TEST_IMAGES = ".data/fer2013_images/PubliTest"
#PRIVATE_TEST_IMAGES = ".data/fer2013_images/PrivateTest"
CSV_FILE = "./data/fer2013/fer2013.csv"

# Create dictionary of target classes
label_dict = {
 0: 'Angry',
 1: 'Disgust',
 2: 'Fear',
 3: 'Happy',
 4: 'Sad',
 5: 'Surprise',
 6: 'Neutral',
}

#Where data will be stored
train_x =[]
train_y = []
test_x = []
test_y = []

#PROCESS_DATA
def split_data():
    global train_x, train_y, test_x, test_y

    dataset = pd.read_csv(CSV_FILE)
    data_type = dataset['Usage']
    pixels = dataset['pixels']
    emotions = dataset['emotion']

    #SPLIT DATA
    print("Spliting data...")
    for i in range(len(dataset)):
        if data_type.iloc[i] == "Training":
            x = []
            for word in pixels[i].split():
                x.append(int(word))
            train_x.append(x)
            train_y.append(emotions.iloc[i])

        else:
            x = []
            for word in pixels[i].split():
                x.append(int(word))
            test_x.append(x)
            test_y.append(emotions.iloc[i])

#SHOWS THE FIRST IMAGE AS PLOT
def print_plot():
    print("Showing a plot...")
    plt.figure(figsize=[5,5])
    plt.subplot(121)
    curr_img = np.reshape(train_x[0], (48,48))
    plt.imshow(curr_img, cmap='gray')
    plt.show()

#PRE-PROCESSING - Images with values from 0 to 1
def pre_processing():
    global train_x, train_y, test_x, test_y

    #CONVERT TO NUMPY
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    #RESHAPE
    train_x = train_x.reshape(-1, 48, 48, 1)
    train_x = train_x.astype('float32')
    train_x /= 255

    test_x = test_x.reshape(-1, 48, 48, 1)
    test_x = test_x.astype('float32')
    test_x /= 255

#DEEP NEURAL NETWORK
def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

     # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=7)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    global train_x, train_y, test_x, test_y

    split_data()
    #To print the first image as a plot uncomment the line below
    #print_plot()

    pre_processing()

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./tmp/first")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    '''
    # Train the model
    print("Training...")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])
    '''

    # Evaluate the model and print results
    print("Evaluating...")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
