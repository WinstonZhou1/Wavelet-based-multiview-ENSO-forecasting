# 2D Convolutional Autoencoder

def dwt_matrix_1d(filter_banks, signdef iiautoencoder(dataset2, ensoerw):
    #sst2 = dataset2[:, i, j, :, :]  # Shape: (timesteps, height, width, channels)
    ensoer = ensoerw
    dataset1 = np.expand_dims(dataset2, axis=-1)
    print(dataset1.shape)
    timesteps, height, width, channels = dataset1.shape
    dataset = dataset1

    scaler = StandardScaler()

    dataset_normalized = np.zeros_like(dataset)
    for t in range(timesteps):

        flat_data = dataset[t].reshape(-1, channels)

        scaled_data = scaler.fit_transform(flat_data)

        dataset_normalized[t] = scaled_data.reshape(height, width, channels)


    input_shape = (height, width, channels)

    input_layer = Input(shape=input_shape)

    # Encoder
    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)

    # Decoder
    decoded = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
    decoded = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
 
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    history = autoencoder.fit(
        dataset_normalized, dataset_normalized,
        epochs=40,
        batch_size=24, 
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Train + Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    encoder = Model(inputs=input_layer, outputs=encoded)
    x_encoded = encoder.predict(dataset_normalized)
    print("ski")
    print(x_encoded.shape)
    x_encoded_reshaped = np.reshape(x_encoded, (x_encoded.shape[0], -1))
    print("bi")
    print(x_encoded_reshaped.shape)
    def hiffest(encoded_data, enso_data):
      encoded_data = encoded_data[11:]
      num_samples = encoded_data.shape[0]

      encoded_flattened = encoded_data.reshape(num_samples, -1)

      encoded_flattened = np.nan_to_num(encoded_flattened)
      correlations = np.array([scipy.stats.pearsonr(encoded_flattened[:, i], enso_data.squeeze())[0] for i in range(56)])
      correlations = np.nan_to_num(correlations) 
        
      highest_corr_index = np.argmax(correlations)
      highest_corr_data = encoded_data[highest_corr_index]
      highest_corr = correlations[highest_corr_index]

      print(f"Highest Pearson correlation: {highest_corr}")

      return highest_corr_data

    #highestcorrdat = hiffest(x_encoded_reshaped, ensoerw)
    return x_encoded
al_length):

