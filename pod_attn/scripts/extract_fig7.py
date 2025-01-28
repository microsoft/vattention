import os
import sys
import re
file_path = sys.argv[1]

tests = {}
iters = 0
with open(file_path, "r") as file:
    for line in file:
        match = re.match(r"Blks: \d+, Iters: (\d+)", line)
        if(match):
            iters = int(match.group(1))
            tests[iters] = {}
        test = re.match(r"(.+): (\d+) us; throughput \d+ GigaOps/s", line)
        if(test):
            name = test.group(1)
            time = int(test.group(2))
            tests[iters][name] = time


for iters in tests:
    print("computeiters", end="\t")
    for test in tests[iters]:
        print(test, end="\t")
    print()
    break
for iters in tests:
    print(iters, end="\t")
    for test in tests[iters]:
        print(tests[iters][test], end="\t")
    print()
