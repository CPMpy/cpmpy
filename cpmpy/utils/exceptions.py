class NullShapeError(Exception):
    def __init__(self, shape, message="Shape should be non-zero"):
        self.shape = shape
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'{self.shape}: {self.message}'