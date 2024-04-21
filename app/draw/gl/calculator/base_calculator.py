from abc import abstractmethod


class BaseCalculator:
    """
    Simple interface for positioning layers on a grid
    Override this to implement positioning algorithm
    For example:
    NumpyCalculator - calculate positions using numpy
    OpenClCalculator - calculate positions using opencl
    ComputeShaderCalculator - calculate positions using opengl shaders
    NumpyCircleCalculator - numpy calculator to produce circle shapes
    OpenClStarCalculator - opencl calculator to produce star shapes
    ...
    """

    @abstractmethod
    def init(self):
        """
        Initialization code if required. For example compiling shaders or activating open CL
        :return:
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Any cleanup code that needs to be done after calculation is completed
        :return:
        """
        pass

    @abstractmethod
    def measure(self, elements_count):
        """
        Total size is always a rectangle
        Imagine a rectangle grid with a shape in it
        Shape is created from elements positioned in calculate_positions (layer data)
        Grid is a rectangle containing the shape (layer container)
        :param elements_count:
        :return: [size_x, size_y]: total size of the grid containing an image
        """
        pass

    @abstractmethod
    def calculate_positions(self, elements_count, start_index, end_index):
        """
        Positions elements on a grid such way they create a specific shape
        For example circle, ellipsis, square, star and more
        :param elements_count:
        :param start_index: start of the batch
        :param end_index: end of the batch
        :return: grid_x, grid_y: the arrays of positions for every [0:elements_count]
        """
        pass
