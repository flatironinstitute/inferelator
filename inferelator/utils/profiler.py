import os, psutil, argparse, time, csv

def main():
    """
    Module that produces a profile of process memory usage and saves it to a file
    """

    ap = argparse.ArgumentParser(description="Create a prior from a genome, TF motifs, and an optional BED file")

    # REQUIRED ARGUMENTS ###############################################################################################

    ap.add_argument("-p", dest="pid", help="Process ID", type=int, required=True)
    ap.add_argument("-o", dest="out", help="Output TSV", metavar="FILE", required=True)
    ap.add_argument("-t", dest="timer", help="Time interval", type=int, default=1)
    args = ap.parse_args()

    pid, t_step = args.pid, args.timer 

    time_start = time.time()
    with open(args.out, mode="w") as out_fh:
        csv_handler = csv.writer(out_fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        csv_handler.writerow(["Time", "Memory"])

        while psutil.pid_exists(pid):
            csv_handler.writerow(['%.3f' % (time.time() - time_start), int(get_current_memory(pid) / 1e6)])
            out_fh.flush()
            time.sleep(t_step)


def get_current_memory(proc_id):
    proc = psutil.Process(proc_id)
    return proc.memory_info().rss


if __name__ == '__main__':
    main()
