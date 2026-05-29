import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, LeakyReLU


def resunet_down_block(input_x, kn1, kn2, kn3, side_kn):
    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # ----- Shortcut branch -----
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = LeakyReLU(negative_slope=0.1)(y)

    # ----- Merge -----
    out = Add()([x, y])
    out = LeakyReLU(negative_slope=0.1)(out)

    return out


def resunet_up_block(input_x, kn1, kn2, kn3, side_kn):
    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # ----- Shortcut branch -----
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = LeakyReLU(negative_slope=0.1)(y)

    # ----- Merge -----
    out = Add()([x, y])
    out = LeakyReLU(negative_slope=0.1)(out)

    return out


def resunet_identity_block(input_x, kn1, kn2, kn3):
    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # ----- Identity merge -----
    out = Add()([x, input_x])
    out = LeakyReLU(negative_slope=0.1)(out)

    return out