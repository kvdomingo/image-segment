class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> tuple:
        return self.x, self.y

    def __repr__(self) -> tuple:
        return self.x, self.y
