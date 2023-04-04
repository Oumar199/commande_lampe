def new_width_after_conv(w: int, f: int, p: int, p_out: int, s: int, transpose: bool):
    """Calculate the new width of the input after applying a convolution layer

    Args:
        w (int): The initial input width
        f (int): The kernel size
        p (int): The padding
        p_out (int): The output padding
        s (int): The stride
        transpose (int): indicate if we had transposed the convolution layer
    """
    if not transpose:
        return int(1 + (w - f + 2*p)/s)
    else:
        return int((w - 1)*s + f - 2*p + p_out)
