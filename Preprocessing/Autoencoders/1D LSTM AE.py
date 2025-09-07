def lstm2_autoencoder(dataset, ensoer):
    #[:, i, j, :, :]  # Shape: (timesteps, height, width)

    sst2 = dataset
    timesteps, height, width = dataset.shape

    flattened_data = dataset.reshape(timesteps, height * width)


    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(flattened_data)

    normalized_data = normalized_data.reshape(timesteps, 1, height * width)  # (timesteps, 1, feature_dim)

    
    input_layer = Input(shape=(1, height * width))

    print("bi1")
 
    encoded = LSTM(128, activation='relu', return_sequences=False)(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)


    decoded = Dense(128, activation='relu')(encoded)
    decoded = RepeatVector(1)(decoded)  
    decoded = LSTM(height * width, activation='sigmoid', return_sequences=True)(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    print("bi")

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    history = autoencoder.fit(
        normalized_data, normalized_data,
        epochs=100,
        batch_size=24,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    print("di")

    encoder = Model(inputs=input_layer, outputs=encoded)
    encoded_data = encoder.predict(normalized_data)
    print(encoded_data.shape)

    def correlation(dataset, target):
      encoded_dat12 = dataset[11:] 
      encoded_dat12 = np.nan_to_num(encoded_dat12)  

      correlations = []
      for i in range(encoded_dat12.shape[1]):
          corr = np.corrcoef(encoded_dat12[:, i], target)[0, 1]

          if corr < -1.0:
              corr = -1.0
          elif corr > 1.0:
              corr = 1.0

          correlations.append(corr)

      correlations = np.array(correlations)

      selected_predictors = encoded_dat12[:, np.abs(correlations) >= 0.8]

      print(f"# predictors selected: {selected_predictors.shape[1]}")
      print(correlations)
      correlations = np.nan_to_num(correlations)

      return correlations


    corr = correlation(encoded_data, ensop)
    highest_corr_index = np.argmax(abs(corr))
    highest_corr_data = encoded_data[:, highest_corr_index]
    print(highest_corr_index)

    highest_corr = corr[highest_corr_index]
    print(f"Highest pearson correlation: {highest_corr}")

    return highest_corr_data
