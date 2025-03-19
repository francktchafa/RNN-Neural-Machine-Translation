import numpy as np
import random
from tqdm import tqdm
from faker import Faker
from babel.dates import format_date
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


# Initialize Faker instance and set seeds for reproducibility
fake = Faker()
SEED = 0
Faker.seed(SEED)
random.seed(SEED)

# Define formats for generating human-readable dates
# FORMATS = [
#     'short', 'medium', 'long', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full',
#     'full', 'd MM YY', 'd MMM YYYY', 'd MMM, YYYY', 'd MMMM YYYY', 'd MMMM, YYYY', 'dd MMM YYYY',
#     'dd, MMM YYYY', 'dd.MM.YY', 'MMMM d YYYY', 'MMMM d, YYYY'
# ]
FORMATS = [
    'short', 'medium', 'long', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full',
    'full', 'd MM YY', 'd MM YY', 'd MMM YYYY', 'd MMM, YYYY', 'd MMMM YYYY', 'd MMMM, YYYY', 'dd MMM YY',
    'dd MMM YYYY', 'dd, MMM YYYY', 'dd.MM.YY', 'MMMM d YYYY', 'MMMM d, YYYY', 'MMMM dd YYYY', 'MMMM dd, YYYY'
]

# Supported locales for generating dates
LOCALES = ['en_US']


def load_date():
    """
    Generates a random date and formats it into human-readable and machine-readable formats.

    The function creates a fake date object, formats it using one of the pre-defined formats
    (randomly chosen), and prepares it for both human-readable and machine-readable use cases.
    If the formatting process fails, the function handles the exception and returns None values.

    Returns:
        tuple: A tuple containing:
            - human_readable (str): A human-readable date string formatted according to the chosen format.
            - machine_readable (str): An ISO 8601 formatted date string (YYYY-MM-DD).
            - dt (datetime.date): The original date object.
    """
    # Generate a random date object
    dt = fake.date_object()

    try:
        # Format the date in a human-readable way using a random format
        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower().replace(',', '')  # Clean and standardize the format

        # Convert the date object to an ISO 8601 machine-readable string
        machine_readable = dt.isoformat()

    except AttributeError as e:
        # Return None if formatting fails
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
    Generates a dataset of date examples and creates vocabularies for human-readable and machine-readable formats.

    Arguments:
        m (int): Number of examples to generate.

    Returns:
        tuple: A tuple containing:
            - dataset (list): List of tuples, where each tuple consists of:
                - human_readable (str): A human-readable date string (e.g., "25th of June, 2009").
                - machine_readable (str): A machine-readable date string in ISO 8601 format (YYYY-MM-DD).
            - human (dict): Vocabulary mapping of characters from human-readable dates to unique integer indices,
                including `<unk>` (unknown) and `<pad>` (padding) tokens.
            - machine (dict): Vocabulary mapping of characters from machine-readable dates to unique integer indices.
            - inv_machine (dict): Reverse vocabulary mapping of indices to characters for machine-readable dates.

    Functionality:
        - Uses the `load_date` function to generate random date examples.
        - Creates vocabularies for both human-readable and machine-readable formats by iterating through examples.
        - Handles unknown and padding tokens in the human vocabulary for preprocessing.
    """

    human_vocab = set()
    machine_vocab = set()
    dataset = []

    for i in tqdm(range(m)):
        h, m, _ = load_date()  # Generate a random date
        if h is not None:
            dataset.append((h, m))  # Append the human and machine-readable pair
            human_vocab.update(tuple(h))  # Update the human-readable vocabulary
            machine_vocab.update(tuple(m))  # Update the machine-readable vocabulary

    # Create vocabularies and mappings
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))  # Map char to index
    inv_machine = dict(enumerate(sorted(machine_vocab)))  # Map indices to characters (machine-readable)
    machine = {v: k for k, v in inv_machine.items()}  # Reverse map characters to indices (machine-readable)

    return dataset, human, machine, inv_machine


def string_to_int(string, length, vocab):
    """
    Converts a string into a list of integers based on a provided vocabulary, with padding or truncation
    to ensure the output matches a specified length.

    Arguments:
        string (str): The input string to be converted (e.g., 'Wed 10 Jul 2007').
        length (int): The desired length of the output.
                      - If the string is longer, it will be truncated.
                      - If the string is shorter, it will be padded.
        vocab (dict): A dictionary mapping characters to their integer indices.
                      Includes special tokens such as '<unk>' for unknown characters and '<pad>' for padding.

    Returns:
        list: A list of integers (size = `length`) representing the positions of the input string's characters
              in the vocabulary. Unknown characters are replaced with the index of '<unk>', and padding is
              applied as necessary.

    Steps:
        1. Normalize the string to lowercase and remove commas for consistency.
        2. Truncate the string if it exceeds the specified length.
        3. Map each character in the string to its corresponding index in the vocabulary.
        4. Pad with '<pad>' tokens if the string is shorter than the desired length.
    """

    # Standardize the string to lowercase and remove commas
    string = string.lower()
    string = string.replace(',', '')

    # Truncate the string if its length exceeds the specified length
    if len(string) > length:
        string = string[:length]

    # Map each character to its corresponding index in the vocabulary
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    # Pad the result with '<pad>' tokens if the string is shorter than the desired length
    if len(rep) < length:
        rep += [vocab['<pad>']] * (length - len(rep))

    return rep


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    """
    Preprocesses the dataset for input into a sequence-to-sequence model by converting text data
    into integer indices and one-hot encoded representations.

    Arguments:
        dataset (list of tuples): A list of (human-readable date, machine-readable date) pairs.
                                  - Example: [("25th of June, 2009", "2009-06-25"), ...]
        human_vocab (dict): Vocabulary mapping characters in the human-readable format
                            to unique integer indices. Includes special tokens like '<unk>' and '<pad>'.
        machine_vocab (dict): Vocabulary mapping characters in the machine-readable format
                              to unique integer indices.
        Tx (int): Maximum length of the human-readable date input (e.g., 30 characters).
                  Longer inputs will be truncated, and shorter inputs will be padded.
        Ty (int): Maximum length of the machine-readable date output (e.g., 10 characters for "YYYY-MM-DD").
                  Shorter outputs will be padded.

    Returns:
        tuple:
            - X (numpy.ndarray): Array of integer-encoded human-readable dates of shape (m, Tx),
                                 where m is the size of the dataset. Characters are mapped to integers
                                 using `human_vocab`, with padding or truncation as necessary.
            - Y (numpy.ndarray): Array of integer-encoded machine-readable dates of shape (m, Ty),
                                 mapped using `machine_vocab`.
            - Xoh (numpy.ndarray): One-hot encoded representations of `X` with shape (m, Tx, len(human_vocab)).
            - Yoh (numpy.ndarray): One-hot encoded representations of `Y` with shape (m, Ty, len(machine_vocab)).
    """

    # Extract human-readable and machine-readable dates from the dataset
    X, Y = zip(*dataset)

    # Convert human-readable dates to integer-encoded arrays
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])

    # Convert machine-readable dates to integer-encoded arrays
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    # One-hot encode human-readable dates
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))

    # One-hot encode machine-readable dates
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh


def preprocess_and_split_data(dataset, human_vocab, machine_vocab, Tx, Ty, validation=True, split_ratio=(0.7, 0.5)):
    """
    Preprocesses the dataset for input into a sequence-to-sequence model by converting text data
    into integer indices and one-hot encoded representations.

    Arguments:
        dataset (list of tuples): A list of (human-readable date, machine-readable date) pairs.
                                  - Example: [("25th of June, 2009", "2009-06-25"), ...]
        human_vocab (dict): Vocabulary mapping characters in the human-readable format
                            to unique integer indices. Includes special tokens like '<unk>' and '<pad>'.
        machine_vocab (dict): Vocabulary mapping characters in the machine-readable format
                              to unique integer indices.
        Tx (int): Maximum length of the human-readable date input (e.g., 30 characters).
                  Longer inputs will be truncated, and shorter inputs will be padded.
        Ty (int): Maximum length of the machine-readable date output (e.g., 10 characters for "YYYY-MM-DD").
                  Shorter outputs will be padded.
        validation (bool): Indicates whether the dataset should be split into CV and test sets. Default is True.
        split_ratio (tuple): Defines the proportions of the dataset allocated to training and validation/test sets.
                     Default is (0.7, 0.5):
                     - 0.7 means 70% of the dataset is used for training.
                     - 0.5 means the remaining 30% is equally divided between validation and test sets,
                       resulting in 15% of the total data for each.


    Returns:
        If `validation=True`:
            - X_train, Y_train: Integer-encoded training data.
            - Xoh_train, Yoh_train: One-hot encoded training data.
            - X_cv, Y_cv: Integer-encoded cross-validation data.
            - Xoh_cv, Yoh_cv: One-hot encoded cross-validation data.
            - X_test, Y_test: Integer-encoded test data.
            - Xoh_test, Yoh_test: One-hot encoded test data.

        If `validation=False`:
            - X, Y: Integer-encoded full dataset.
            - Xoh, Yoh: One-hot encoded full dataset.
    """

    # Extract human-readable and machine-readable dates from the dataset
    X, Y = zip(*dataset)

    # Convert human-readable dates to integer-encoded arrays
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])

    # Convert machine-readable dates to integer-encoded arrays
    Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])

    if validation:
        X_train, X_, Y_train, Y_ = train_test_split(X, Y, train_size=split_ratio[0],
                                                    shuffle=True, random_state=SEED)
        X_cv, X_test, Y_cv, Y_test = train_test_split(X_, Y_, train_size=split_ratio[1],
                                                      shuffle=True, random_state=SEED)

        # One-hot encode human-readable dates
        Xoh_train = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X_train)))
        Xoh_cv = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X_cv)))
        Xoh_test = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X_test)))

        # One-hot encode machine-readable dates
        Yoh_train = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y_train)))
        Yoh_cv = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y_cv)))
        Yoh_test = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y_test)))

        return X_train, Y_train, Xoh_train, Yoh_train, X_cv, Y_cv, Xoh_cv, Yoh_cv, X_test, Y_test, Xoh_test, Yoh_test

    else:
        # One-hot encode human-readable dates on the full dataset
        Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))

        # One-hot encode machine-readable dates
        Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

        return X, np.array(Y), Xoh, Yoh


def softmax(x, axis=1):
    """
    Applies the softmax activation function to the input tensor.

    Args:
        x (Tensor): Input tensor to which the softmax function will be applied.
        axis (int): Axis along which the softmax normalization is performed. Default is 1.

    Returns:
        Tensor: A tensor with the same shape as the input, where the values are the
                result of the softmax transformation.

    Raises:
        ValueError: If the input tensor `x` is 1-dimensional. The softmax function
                    requires at least two dimensions to operate along the specified axis.

    Notes:
        - For a 2D tensor, the Keras backend's `K.softmax()` function is used directly for efficiency.
        - For tensors with more than 2 dimensions, the function computes the softmax manually:
          1. Exponentiates each element after subtracting the maximum value along the axis
             (to improve numerical stability).
          2. Divides each element by the sum of all exponentiated values along the axis.
    """

    ndim = K.ndim(x)  # Determine the number of dimensions in the input tensor

    if ndim == 2:
        # Use Keras's built-in softmax for 2D tensors
        return K.softmax(x)
    elif ndim > 2:
        # Manually compute softmax for tensors with more than 2 dimensions
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))  # Exponentiation with stability adjustment
        s = K.sum(e, axis=axis, keepdims=True)             # Sum along the specified axis
        return e / s
    else:
        # Raise an error for 1D tensors
        raise ValueError('Cannot apply softmax to 1-dimensional tensors.')


def plot_history(history):
    """
    Plots training and validation loss and accuracy from the history object.

    Arguments:
        history: Keras History object returned by model.fit().
    """
    # Extract data from the history object
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_accuracy(model, datasets, calc_loss=True):
    """
    Calculates and prints accuracy for training, validation, and test datasets.

    Arguments:
        model: Trained Keras model.
        datasets: A dictionary containing input and output data for training, validation, and test sets.
                  Example structure:
                  {
                      "Training": (Xoh_train, s0_train, c0_train, outputs_train),
                      "Validation": (Xoh_cv, s0_cv, c0_cv, outputs_cv),
                      "Test": (Xoh_test, s0_test, c0_test, outputs_test)
                  }
        calc_loss: Bool that determines whether to calculate the loss. DEfault is True
    Returns:
        A dictionary with calculated accuracies for each dataset.
    """

    accuracies = {}
    losses = {}

    for label, (Xoh, s0, c0, outputs) in datasets.items():
        # Predict the sequences
        predictions = model.predict([Xoh, s0, c0], verbose=0)
        predicted_sequences = np.argmax(predictions, axis=-1)  # Convert probabilities to indices (Ty, m)
        true_sequences = np.argmax(outputs, axis=-1)  # True indices (Ty, m)

        # Calculate correctness for each sequence
        correct = np.all(predicted_sequences == true_sequences, axis=0)  # Shape: (m,)
        accuracy = np.mean(correct)  # Overall accuracy

        # Store and print the accuracy
        accuracies[label] = accuracy * 100  # In percentage
        print(f"{label} Dataset \n Accuracy: {accuracy * 100:.2f}%")

        if calc_loss:
            results = model.evaluate([Xoh, s0, c0], outputs, batch_size=len(Xoh), verbose=0)
            overall_loss = results[0]
            step_losses = np.array(results[1:len(true_sequences+1)])
            step_accuracies = np.array(results[len(true_sequences)+1:])

            # Print timestep losses in scientific notation
            print(" Timestep Losses:", np.array2string(step_losses, formatter={'float_kind': lambda x: f"{x:.3e}"}))
            print(f" Timestep Accuracies: {np.round(step_accuracies, decimals=3)} \n")

    return accuracies, losses


def get_incorrect_indices(model, Xoh, Yoh, s0, c0, label, vocab, print_opt=True):
    """
    Identifies the indices of incorrectly labeled samples.

    Arguments:
        model: Keras trained model for making predictions
        Xoh: One-hot encoded input data (e.g., X_train)
        Yoh: One-hot encoded true labels (e.g., Yoh_train)
        s0: Initial hidden state for the LSTM
        c0: Initial cell state for the LSTN
        label: Label of the dataset (e.g., 'Training')
        vocab: Dictionary of the reverse vocabulary mapping of indices to characters for machine-readable dates.
        print_opt: Print the mismatched items. Default is True

    Returns:
        incorrect_indices: List of indices where predictions do not match true labels.
    """

    # Convert true labels to class indices
    true_sequences = np.argmax(Yoh, axis=-1)

    # Get model predictions and convert to class indices
    out = model.predict([Xoh, s0, c0], verbose=0)
    out = np.swapaxes(out, 0, 1)  # Swap axes if needed
    pred_sequences = np.argmax(out, axis=-1)

    # Ensure shapes match
    # print(f"True shape: {true_sequences.shape}, Predicted shape: {pred_sequences.shape}")
    assert true_sequences.shape == pred_sequences.shape, "The true and predicted shape do not match"

    # Identify incorrect indices
    incorrect_indices = []
    for i, (true, pred) in enumerate(zip(true_sequences, pred_sequences)):
        if not np.array_equal(true, pred):  # Find mismatches
            incorrect_indices.append(i)

    print(f"\n{label} dataset incorrectly labeled samples: {len(incorrect_indices)} out of {Xoh.shape[0]} "
          f"({len(incorrect_indices) / Xoh.shape[0] * 100:.2f}% error rate)")

    if print_opt and len(incorrect_indices) > 0:
        print(f"Examples of incorrectly labeled samples in the {label.lower()} dataset")

        for j, ind in enumerate(incorrect_indices):
            if j < 10:  # only print a maximum of 10 incorrectly labeled samples
                source = [vocab[int(i)] for i in true_sequences[ind]]
                output = [vocab[int(i)] for i in pred_sequences[ind]]

                print(" Source:", ''.join(source),
                      " Output:", ''.join(output))

    return incorrect_indices


def int_to_string(ints, inv_vocab):
    """
    Converts a list of vocabulary indexed to machine-readable characters

    Arguments:
        ints -- List of integers representing indexes in the machine's vocabulary
        inv_vocab -- Pyton dictionary mapping indexes to characters

    Returns:
        l -- list of characters corresponding to the given indixes.
    """

    the_list = [inv_vocab[i] for i in ints]

    return the_list


def plot_attention_map(modelx, Tx, Ty, input_vocabulary, inv_output_vocabulary, sample_text, n_s):
    """
    Visualize the attention map of the sequence to sequence model with attention.

    This function calculates and plots the attention map for a given input text using an attention-based
    sequence-to-sequence (Seq2Seq) model. It shows how much "attention" the model places on each input timestep
    when predicting each output timestep.

    Arguments:
        modelx (keras.Model): The trained model with an attention layer.
        Tx (int): The maximum input sequence length.
        Ty (int): The maximum output sequence length.
        input_vocabulary (dict): A dictionary mapping input text characters to integer indices.
        inv_output_vocabulary (dict): A dictionary mapping integer indices to output text characters.
        sample_text (str): The input text (sequence) for which the attention map is generated.
        n_s (int): The size of the decoder's hidden state.

    Returns:
        attention_map (np.ndarray): A 2D array of shape (Ty, Tx) representing the normalized attention weights.

    Workflow:
        1. Extracts the attention weights for each output timestep from the attention layer of the model.
        2. Preprocesses the input text by encoding it into integer indices and then one-hot encoding.
        3. Computes the attention map for all output timesteps by normalizing the attention weights.
        4. Predicts the output text using the model and decodes the predicted text.
        5. Plots the attention map as a heatmap with input and output sequences labeled.
    """

    attention_map = np.zeros((Ty, Tx))
    layer = modelx.get_layer('attention_weights')  # Extract the attention layer

    # Model for generating attention weights for all decoding timesteps
    f = Model(modelx.inputs, [layer.get_output_at(t) for t in range(Ty)])

    # Initialize the decoder's hidden and cell states
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))

    # Encode the input text
    encoded = np.array(string_to_int(sample_text, Tx, input_vocabulary)).reshape((1, Tx))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    # Get attention weights for all decoding timesteps
    alphas = f([encoded, s0, c0])
    # print(np.array(alphas).shape)

    # Compute the attention map
    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = alphas[t][0, t_prime]  # Extract attention weights

    # Normalize attention map
    row_max = attention_map.max(axis=1).reshape(-1, 1)
    attention_map = attention_map / row_max

    # Predict the output text
    prediction = modelx.predict([encoded, s0, c0])

    # Decode the predicted text
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)))

    predicted_text = list(predicted_text)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)
    text = list(sample_text)

    # Get lengths of input and output sequences
    input_length = len(sample_text)
    output_length = Ty

    attention_map = attention_map[:, :input_length]

    # Plot the attention map
    plt.clf()  # Clear any previous plot
    fig, ax = plt.subplots()

    # Add attention map as an image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Greys_r')

    # Add colorbar
    cbar = fig.colorbar(i, ax=ax, orientation='horizontal', pad=0.2)
    cbar.ax.set_xlabel('Alpha values from the "attention_weights" layer', labelpad=2)

    # Add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])
    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text[:input_length], rotation=45)

    # Label axes
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # Add grid
    ax.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Display the plot
    plt.show()

    return attention_map
