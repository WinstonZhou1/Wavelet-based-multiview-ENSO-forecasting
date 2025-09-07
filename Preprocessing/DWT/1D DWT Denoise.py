#still uses 4b 4r wavelet filter bank

def dwt_matrix_1d(filter_banks, signal_length):
    num_bands = len(filter_banks)
    matrix_size = num_bands * (signal_length // num_bands) 
    res = np.zeros((matrix_size, signal_length))

    shift = num_bands
    row = 0
    for filter_bank in filter_banks:
        current_offset = 0
        for _ in range(signal_length // num_bands): 
            for (i, value) in enumerate(filter_bank):
                col = (current_offset + i) % signal_length
                res[row][col] = value
            current_offset += shift
            row += 1
    return res

def dwt_1d(signal):
    padded_signal = np.pad(signal, (0, (4 - len(signal) % 4) % 4), 'constant')

    matrix = dwt_matrix_1d(wv44, len(padded_signal))
    transformed_signal = np.dot(matrix, padded_signal)

    transformed_bands = np.split(transformed_signal, 4)

    return transformed_bands

def wavelet_denoising(S):
    n = S.shape[0] # Number of values
    if n % 4 != 0:
      # Correct the padding for a 1D array
      S = np.pad(S, (0, 4 - (n % 4)), mode='constant')
    k = S.shape[0] // 4
    W = dwt_matrix(wv44, k)
    WS = W @ S # Wavelet domain
    for i in [1, 2, 3]:
      start_idx = i * k
      end_idx = (i + 1) * k
      D = WS[start_idx:end_idx]
      cutoff = np.std(D) * np.sqrt(2 * np.log(k))
      D[np.abs(D) < cutoff] = 0
      WS[start_idx:end_idx] = D
    S_denoised = (W.T @ WS)[:n]
    return S_denoised
