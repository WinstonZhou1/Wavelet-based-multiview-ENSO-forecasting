# Filter bank for the 4-band 4-regular wavelet - higher resolution image/dataset
wv44 = (
  ( 0.0857130200,  0.1931394393,  0.3491805097,  0.5616494215,  0.4955029828,  0.4145647737,  0.2190308939, -0.1145361261,
   -0.0952930728, -0.1306948909, -0.0827496793,  0.0719795354,  0.0140770701,  0.0229906779,  0.0145382757, -0.0190928308),
  (-0.1045086525,  0.1183282069, -0.1011065044, -0.0115563891,  0.6005913823, -0.2550401616, -0.4264277361, -0.0827398180,
    0.0722022649,  0.2684936992,  0.1691549718, -0.4437039320,  0.0849964877,  0.1388163056,  0.0877812188, -0.1152813433),
  ( 0.2560950163, -0.2048089157, -0.2503433230, -0.2484277272,  0.4477496752,  0.0010274000, -0.0621881917,  0.5562313118,
   -0.2245618041, -0.3300536827, -0.2088643503,  0.2202951830,  0.0207171125,  0.0338351983,  0.0213958651, -0.0280987676),
  ( 0.1839986022, -0.6622893130,  0.6880085746, -0.1379502447,  0.0446493766, -0.0823301969, -0.0923899104, -0.0233349758,
    0.0290655661,  0.0702950474,  0.0443561794, -0.0918374833,  0.0128845052,  0.0210429802,  0.0133066389, -0.0174753464)
)
def dwt_matrix(filter_banks, rows_per_filter_bank):
    filter_banks = np.array(filter_banks)
    num_bands = len(filter_banks)

    matrix_size = num_bands * rows_per_filter_bank
    res = np.zeros((matrix_size, matrix_size))

    shift = num_bands
    row = 0
    for filter_bank in filter_banks:
        current_offset = 0
        for _ in range(rows_per_filter_bank):
            for (i, value) in enumerate(filter_bank):
                res[row][(current_offset + i) % matrix_size] = value
            current_offset += shift
            row += 1
    return res

num_bands = len(wv44)
signal_shape = None
matrix1 = None
matrix2 = None

def initialize_wavelet_matrices(signal_shape):
    global matrix1, matrix2
    x, y = signal_shape
    k1 = x // num_bands
    k2 = y // num_bands
    matrix1 = dwt_matrix(wv44, k1)
    matrix2 = dwt_matrix(wv44, k2)

def dwt_2d(signal):
    global matrix1, matrix2, signal_shape
    if signal_shape != signal.shape:
        signal_shape = signal.shape
        initialize_wavelet_matrices(signal_shape)


    values = np.einsum('ij,jk->ik', matrix1, signal)
    transformed_signal = np.einsum('ik,kj->ij', values, matrix2)


    transformed_blocks = np.array(tuple(np.split(row, num_bands, axis=1) for row in np.split(transformed_signal, num_bands, axis=0)))
    return transformed_blocks
