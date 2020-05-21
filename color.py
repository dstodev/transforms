import numpy as np
from matplotlib import image, pyplot

from src.color import Color

if __name__ == "__main__":
    trees = "resource/Larix_decidua_Aletschwald.jpg"
    image = np.array(image.imread(trees), dtype=float)

    np.apply_along_axis(Color.shift(0.8, 0.5, 1.2), 2, image)

    image = image.astype(int)

    print(image)
    pyplot.imshow(image)
    pyplot.show()
