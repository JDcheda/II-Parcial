import math

class Figura:
    def __init__(self, radio):
        self.radio = radio

    def area_circulo(self):
        return math.pi * self.radio ** 2

    def area_esfera(self):
        return 4 * math.pi * self.radio ** 2

    def volumen_esfera(self):
        return (4/3) * math.pi * self.radio ** 3
