
import numpy as np

def delay_embed(data, embedding_dimension=20, time_delay=1):
    """Delay embed data by concatenating consecutive increase delays.
    Parameters
    ----------
    data : array, 1-D
        Data to be delay-embedded.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.
    Returns
    -------
    x : array, 2-D (samples x dim)
        Delay embedding reconstructed data in higher dimension.
    """
    if type(time_delay) is not int:
        time_delay = int(time_delay)

    num_samples = len(data) - time_delay * (embedding_dimension - 1)
    return np.array([data[dim * time_delay:num_samples + dim * time_delay] for dim in range(embedding_dimension)]).T[:,::-1]

def hessian(x):
    """
    Calculates the second derivative of a given input.

    Parameters:
    x (numpy array): The input data.

    Returns:
    numpy array: The second derivative of the input data.
    """
    return np.gradient(np.gradient(x, axis=0), axis=0)


def calculate_variance_2nd_derivative(time_series, embedding_dimension=3, time_delay=1):
    """
    Calculate the variance of the second derivatives along a reconstructed phase space trajectory
    from a time series using Time Delay Embedding technique.

    Parameters:
    time_series (numpy array): The input time series data.
    embedding_dimension (int): The number of consecutive values in each row (default 3).
    time_delay (int): The number of time steps to shift each row (default 1).

    Returns:
    float: The variance of the second derivatives.
    """
    # Phase space embedding
    matrix = delay_embed(time_series, embedding_dimension, time_delay)
    #print(time_series)
    #print(matrix)


    # Second derivatives using Hessian
    second_derivatives = hessian(matrix)
    #print(second_derivatives)
    #sys.exit()


    # Squaring, summing for each point, square root
    summed_squared = np.sqrt(np.sum(np.square(second_derivatives), axis=1))

    # Variance
    variance = np.var(summed_squared)

    return variance


