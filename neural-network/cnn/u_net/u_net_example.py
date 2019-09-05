# class UNET(object):
    def __init__(self, img_rows=256, img_cols=256, channel=3, n_filters=16, dropout=0.1, batchnorm=True ):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.model = None
        print('Input shape : (%i, %i, %i)' % (self.img_rows, self.img_cols, self.channel))

    def conv2d_block(self, input_tensor, filters, kernel_size=3):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def unet(self):

        input_img = Input((self.img_rows, self.img_cols, self.channel))

        # Contracting Path
        c1 = self.conv2d_block(input_img, filters=self.n_filters * 1, kernel_size=3)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout)(p1)

        c2 = self.conv2d_block(p1, filters=self.n_filters * 2, kernel_size=3)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)

        c3 = self.conv2d_block(p2,filters=self.n_filters * 4, kernel_size=3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)

        c4 = self.conv2d_block(p3, filters=self.n_filters * 8, kernel_size=3)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)

        c5 = self.conv2d_block(p4, filters=self.n_filters * 16, kernel_size=3)

        # Expansive Path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = self.conv2d_block(u6, filters=self.n_filters * 8, kernel_size=3)

        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = self.conv2d_block(u7, filters=self.n_filters * 4, kernel_size=3)

        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = self.conv2d_block(u8, filters=self.n_filters * 2, kernel_size=3)

        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(self.dropout)(u9)
        c9 = self.conv2d_block(u9, filters=self.n_filters * 1, kernel_size=3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        self.model = Model(inputs=[input_img], outputs=[outputs])

        # self.model.summary()
        # plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return self.model
