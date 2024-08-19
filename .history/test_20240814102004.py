import inspect

def print_with_line_number(message):
    line_number = inspect.currentframe().f_back.f_lineno
    print(f"[Line {line_number}]: {message}")

print_with_line_number("Hello, World!")
print_with_line_number("This is a test message.")