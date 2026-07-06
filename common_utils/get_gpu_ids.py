import os
import argparse

def str2float(string):
    string = ''.join(list(filter(lambda ch: ch in '0123456789.', string)))
    if string == '':
        return None
    else:
        return float(string)


def get_gpu_info():
    '''
    ## Description:
        The function that get the message from nvidia-smi and returns the dict of gpu information.
    ## Return:
        gpu_status_dict: dict
    '''
    gpu_status_dict = dict()

    # Use stable CSV query interface instead of screen-scraping text output
    query = 'index,memory.used,memory.total,utilization.gpu,fan.speed,temperature.gpu,power.draw,power.limit'
    output = os.popen(
        f'nvidia-smi --query-gpu={query} --format=csv,noheader,nounits'
    ).read().strip()

    for line in output.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 8:
            continue
        gpu_id = parts[0]
        gpu_status_dict[gpu_id] = {
            'fan': str2float(parts[4]),
            'temperature': str2float(parts[5]),
            'used power': str2float(parts[6]),
            'total power': str2float(parts[7]),
            'used memory': str2float(parts[1]),
            'total memory': str2float(parts[2]),
            'gpu util': str2float(parts[3]),
            'process num': 0,
        }

    # Count compute processes per GPU via pmon (falls back gracefully if unavailable)
    pmon_output = os.popen('nvidia-smi pmon -c 1 -s u 2>/dev/null').read().strip()
    for line in pmon_output.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] != '-':
            gpu_id = parts[0]
            if gpu_id in gpu_status_dict:
                gpu_status_dict[gpu_id]['process num'] += 1

    return gpu_status_dict


def get_args():
    parser = argparse.ArgumentParser(description='The program that get ids of usable GPUs. ')
    parser.add_argument('-N', '--num', type=int, help="The number of GPUs required. ")
    parser.add_argument('-M', '--memory', type=int, default=None, help="The minimum video memory required per GPU. ")
    parser.add_argument('-P', '--process', type=int, default=None, help="The maximum number of running processes per GPU. ")
    parser.add_argument('-U', '--util', type=int, default=None, help="The maximum utilization rate per GPU. ")
    parser.add_argument('-I', '--id', nargs='*', default=None, help="The list of specified GPU id(s).")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print per-GPU rejection reasons to stderr when no GPU qualifies.")

    return parser.parse_args()


def main():
    import sys
    args = get_args()
    assert args.num is not None and args.num>=0, "[ERROR] The number of GPUs required is None/wrong. "
    usable_gpu_list = list()
    gpu_status_dict = get_gpu_info()
    rejection_reasons = {}
    for id in gpu_status_dict.keys():
        if args.id is not None and id not in args.id:
            continue
        reasons = []
        info = gpu_status_dict[id]
        free_mem = (info['total memory'] or 0) - (info['used memory'] or 0)
        if args.memory is not None:
            if info['used memory'] is None or info['total memory'] is None:
                reasons.append('memory N/A')
            elif free_mem < args.memory:
                reasons.append(f'free mem {free_mem:.0f}/{args.memory} MB')
        if args.process is not None:
            if info['process num'] is None:
                reasons.append('process count N/A')
            elif info['process num'] > args.process:
                reasons.append(f'procs {info["process num"]}>{args.process}')
        if args.util is not None:
            if info['gpu util'] is None:
                reasons.append('util N/A')
            elif info['gpu util'] > args.util:
                reasons.append(f'util {info["gpu util"]:.0f}%>{args.util}%')
        if reasons:
            rejection_reasons[id] = ', '.join(reasons)
        else:
            usable_gpu_list.append(id)

    if len(usable_gpu_list) >= args.num:
        print(','.join(usable_gpu_list[:args.num]))
    elif args.verbose:
        lines = [f'  GPU {id}: {reason}' for id, reason in sorted(rejection_reasons.items(), key=lambda x: int(x[0]))]
        print('No qualifying GPU found:\n' + '\n'.join(lines), file=sys.stderr)


if __name__ == '__main__':
    main()