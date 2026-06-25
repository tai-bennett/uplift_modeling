SAVE_FORMAT = "png"
IMAGE_SCALE = 2


def get_numeric_columns():
    pass


def get_category_columns():
    pass


def save_figure(fig, root, name, fmt=SAVE_FORMAT):
    path = root / (name + "." + fmt)
    fig.write_image(str(path), scale=IMAGE_SCALE)
