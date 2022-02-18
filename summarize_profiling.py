import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

def operate(i, line, tag, times, start_times, end_to_end_durations, durations, len_tags):
    if not line.startswith(tag):
        return

    rest = line[len(tag) + 1:].rstrip()
    line_id = rest[:rest.find(' ')]
    line_time = rest[rest.find(' ') + 1:]

    line_time = datetime.strptime(line_time, "%Y-%m-%d %H:%M:%S.%f")

    if i != 0:
        durations[tag].append((line_time - times[line_id]).total_seconds())
    
    times[line_id] = line_time

    if i == 0:
        start_times[line_id] = line_time
    
    if i == len_tags - 1:
        end_to_end_durations[tag].append((line_time - start_times[line_id]).total_seconds())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default = 'profiling.txt', help='profile file')
    args = parser.parse_args()

    iter_tags = ["starting training sample", "finished forward pass", "finished loss", "finished backward", \
    "finished optimizer step", "finished training sample"]
    data_tags = ["entering get item", "finished loading from disk", "finished add front aug", \
    "finished get_bbox and selecting obj", "finished first img_masked", "finished second img_masked", \
    "finished third img_masked", "finished doing densefusion's stuff", "finished my rotation stuff", "finished projecting depth", \
    "finished sampling points from roi", "finished computations"]

    all_tags = iter_tags + data_tags

    start_times_iter = {}
    start_times_data = {}
    times_iter = {}
    times_data = {}

    end_to_end_durations = {}
    durations = {}

    for tag in all_tags:
        end_to_end_durations[tag] = []
        durations[tag] = []

    #format of print's are "tag <id> time"
    with open(args.profile, 'r') as f:
        for line in f:
            for i, tag in enumerate(iter_tags):
                operate(i, line, tag, times_iter, start_times_iter, end_to_end_durations, durations, len(iter_tags))

            for i, tag in enumerate(data_tags):
                operate(i, line, tag, times_data, start_times_data, end_to_end_durations, durations, len(data_tags))

    print("intervals")

    for tag in all_tags:
        tag_durations = durations[tag]
        #0th, so 
        if len(tag_durations) == 0:
            continue
        print(tag, np.mean(tag_durations))

    end_to_end_iter = end_to_end_durations[iter_tags[-1]]
    end_to_end_data = end_to_end_durations[data_tags[-1]]

    print("end to ends")
    print(iter_tags[-1], np.mean(end_to_end_iter))
    print(data_tags[-1], np.mean(end_to_end_data))
                    



if __name__ == "__main__":
    main()