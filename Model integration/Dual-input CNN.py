def make_model(input_shape=(12, 192, 288, 10), non_image_shape=(12,1), output_months=1,use_lstm=True,
               cnn_activation='relu', lstm_activation='tanh', filters_multiplier=16,
               kernel_size=(3, 3), dropout_rate=0.2):


    ConvLayer = lambda filters: Conv2D(filters, kernel_size, padding='same', activation=cnn_activation)
    t = (lambda l: TimeDistributed(l)) if use_lstm else lambda l: l

    image_input = Input(shape=input_shape)
    non_image_input = Input(shape=non_image_shape)

 
    conv1 = t(ConvLayer(1 * filters_multiplier))(image_input)
    pool1 = t(MaxPool2D((2, 2)))(conv1)
    conv2 = t(ConvLayer(2 * filters_multiplier))(pool1)
    pool2 = t(MaxPool2D((2, 2)))(conv2)

    # ConvLSTM2D
    conv3 = ConvLSTM2D(4 * filters_multiplier, kernel_size, padding='same',
                       activation=lstm_activation, return_sequences=False)(pool2)
    conv3 = Dropout(dropout_rate)(conv3)  
    image_features = Flatten()(conv3)

    non_image_input_reshaped = tf.keras.layers.Reshape((non_image_shape[0] * non_image_shape[1],))(non_image_input)
    non_image_dense = tf.keras.layers.Dense(64, activation='relu')(non_image_input_reshaped)


    repeated_non_image_features = tf.keras.layers.RepeatVector(image_features.shape[1])(non_image_dense)

    repeated_non_image_features = Reshape((image_features.shape[1], 64))(repeated_non_image_features)

    image_features = Reshape((image_features.shape[1], 1))(image_features) 

    concatenated_features = tf.keras.layers.Concatenate(axis=-1)([image_features, repeated_non_image_features])

    dense1 = Dense(128, activation='relu')(concatenated_features)
    dense1 = Dropout(dropout_rate)(dense1)  

    # Output layer
    output_layer = Dense(output_months, activation='linear')(dense1)

    model = Model(inputs=[image_input, non_image_input], outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics = ['accuracy'])

    return model

if noni_train.shape != (468, 12, 1):
    noni_train = noni_train.reshape(468, 12, 1)

model00 = make_model(input_shape=(12, 192, 288, 9), non_image_shape=(12,1))
