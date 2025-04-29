import argparse
import ast

class ParseListOfLists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            result = ast.literal_eval(values)
            if not all(isinstance(i, list) and len(i) == 2 for i in result):
                raise ValueError("Each sublist must contain exactly two elements.")
            setattr(namespace, self.dest, result)
        except ValueError as ve:
            raise argparse.ArgumentTypeError(f"Input error: {ve}")
        except:
            raise argparse.ArgumentTypeError("Input should be a list of lists with two strings each (e.g., [['11', '0'], ['13', '0']])")

parser = argparse.ArgumentParser(description="Process a list of lists.")
parser.add_argument("--list_of_heads", type=str, action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")

args = parser.parse_args()
print("Received list of lists:", args.list_of_heads)

list_of_heads = [[int(head[0]), int(head[1])] for head in args.list_of_heads]
print("Parsed list of lists:", list_of_heads)


if __name__ == "__main__":
    # Your script logic here
    pass
