#! /usr/bin/python3
import sys
import argparse

levelupList = lambda l: [
    li | lj
    for i, li in enumerate(l)
    for lj in l[i + 1 :]
    if list(li)[:-1] == list(lj)[:-1]
]


def invoiceCalc(inhands, target, exep=None, max_first=True, num=3):
    inhands.sort()
    step = 0.001
    for i in range(len(inhands) - 1):
        if inhands[i] == inhands[i + 1]:
            inhands[i] += step
            step += 0.001
        else:
            step = 0.001

    if exep != None:
        inhands = {float("%.3f" % x) for x in inhands}
        inhands = inhands - set(exep)

    inhands = list(inhands)
    if inhands == []:
        print("Nothing left.")
        return None
    if target <= min(inhands) or len(inhands) < 2:
        print(min(inhands), "\t", min(inhands))
        return
    if target > sum(inhands) - min(inhands):
        print(tuple(inhands), "\t", sum(inhands))
        return

    if max_first == True:
        max_inhands = max(inhands)
        target -= max_inhands
        inhands.remove(max_inhands)
        if target <= 0:
            print("Max bill in hand is :", max_inhands, "\tResult:", max_inhands)
            return

    inhands = [frozenset([x]) for x in inhands]
    r = {}
    while inhands != []:
        inhandsVal = {row: sum(list(row)) for row in inhands}
        r.update(inhandsVal)
        inhands = [x for x in inhandsVal if inhandsVal[x] < target]
        inhands = levelupList(inhands)

    rv = [x for x in sorted(r.values()) if x >= target][:num]
    r = {x: r[x] for y in rv for x in r if r[x] == y}

    if max_first == True:
        r = {x | frozenset([max_inhands]): y + max_inhands for x, y in r.items()}

    print("= = = = = =")
    for x, y in r.items():
        print("  Choice: ", tuple(x), "\t Result: %.3f" % y)
    print("= = = = = =")
    # return r


# inhands = [146, 86, 74, 146, 128, 280, 70, 136, 89.68, 128, 96, 86, 146, 146, 57, 552, 41.4, 13.85]
# target = 600


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "inhands", nargs="+", type=str, help="Bills that owned, format 1, 2, 3 or 1 2 3"
    )
    parser.add_argument(
        "-t", "--target", required=True, type=float, help="Target number that is needed"
    )

    parser.add_argument(
        "-e",
        "--exception",
        nargs="*",
        type=str,
        default=None,
        help="Execption bills from those owned",
    )
    parser.add_argument(
        "--not_max_first", action="store_true", help="Disable using max bills first"
    )
    parser.add_argument(
        "-o",
        "--output_num",
        type=int,
        default=3,
        help=("Possible result output number"),
    )

    args = parser.parse_args(argv)
    args.inhands = [float(ss.replace(",", "")) for ss in args.inhands]
    if args.exception != None:
        args.exception = [float(ss.replace(",", "")) for ss in args.exception]
    args.max_first = not args.not_max_first

    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    invoiceCalc(
        args.inhands, args.target, args.exception, args.max_first, args.output_num
    )
