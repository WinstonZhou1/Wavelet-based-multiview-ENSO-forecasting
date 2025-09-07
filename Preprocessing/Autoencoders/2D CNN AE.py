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
