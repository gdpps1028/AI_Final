import argparse
from Character_Only import ch_main
from Radical_Character import rd_main
from Stroke_Character import st_main

parser = argparse.ArgumentParser(description="A sample script using argparse")

parser.add_argument('-m', '--Model', type=str, help='C for Character, R for Radical, S for Stroke', default='C', choices=['C', 'R', 'S'])
parser.add_argument('-t', '--Train', action='store_true', help='Add this flag if you want to retrain the model')
parser.add_argument('-e', '--Eval', type=int, help='Test the model, input an interger for amount of testcases', default=100)

args = parser.parse_args()

print(f"Model : {args.Model}")
print(f"Train : {args.Train}")
print(f"Eval  : {args.Eval}")

if __name__ == "__main__":
    if args.Model == 'C':
        ch_main.main(args.Train, args.Eval)
    elif args.Model == 'R':
        rd_main.main(args.Train, args.Eval) 
    elif args.Model == 'S':
        st_main.main(args.Train, args.Eval) 