# Test file to verify Black formatting works
def poorly_formatted_function(x, y, z):
    if x > 0:
        result = x + y * z
        return result
    else:
        return 0


# This should be reformatted by Black
class BadlyFormattedClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
