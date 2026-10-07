"""Live view of training progress. Read-only: it just parses the logs the training scripts write.

Usage:  python watch_training.py            (refreshes every 10 s, Ctrl+C to quit)
        python watch_training.py --once     (print once and exit)
        python watch_training.py --run imnet --last 25
"""
import argparse, glob, json, os, re, subprocess, time

D = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(D, 'logs')
EPOCH = re.compile(r'(\d+)/\d+ - (\d+)s - loss: ([\d.]+) - accuracy: ([\d.]+) - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)')
PATIENCE = 20


def parse_cnn(path):
    txt = open(path, encoding='utf-8', errors='ignore').read()
    rows = [dict(sec=int(m[2]), loss=float(m[3]), acc=float(m[4]), vloss=float(m[5]), vacc=float(m[6]))
            for m in EPOCH.finditer(txt)]
    if 'Traceback' in txt:  # (TF's harmless "ptxas ... Error code: 2" lines are not crashes)
        status = 'CRASHED (see log)'
    elif 'B_val macro-F1' in txt:
        status = 'DONE  ' + re.search(r'B_val macro-F1 [\d.]+', txt)[0]
    elif time.time() - os.path.getmtime(path) > 1800:
        status = 'STALLED? (no output for 30+ min)'
    else:
        status = 'running'
    return rows, status


def bar(v, lo, hi, width=30):
    n = 0 if hi == lo else int(round((v - lo) / (hi - lo) * width))
    return '#' * max(n, 1)


def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
        u, mu, mt, pw, t = [s.strip() for s in out.split(',')]
        return f'GPU {u}% util | {mu}/{mt} MiB | {pw} W | {t} C'
    except Exception:
        return 'GPU: nvidia-smi not available'


def trials(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return f'{name}: none yet'
    recs = [json.loads(l) for l in open(p) if l.strip()]
    if not recs:
        return f'{name}: none yet'
    best = max(recs, key=lambda r: r['macro_mean'])
    last = recs[-1]
    return (f"{name}: {len(recs)} trials | last {last['name']} {last['macro_mean']:.4f} | "
            f"best {best['name']} {best['macro_mean']:.4f} ± {best['macro_std']:.4f}")


def show(a):
    logs = sorted(glob.glob(os.path.join(LOGS, 'cnn_*.log')), key=os.path.getmtime)
    if a.run:
        logs = [l for l in logs if os.path.basename(l) == f'cnn_{a.run}.log'] or logs
    lines = [time.strftime('%H:%M:%S') + '  ' + gpu(), '']
    lines.append(f"{'CNN run':10s} {'ep':>4s} {'best vloss':>10s} {'@ep':>4s} {'wait':>6s} {'s/ep':>5s}  status")
    for p in logs:
        rows, status = parse_cnn(p)
        tag = os.path.basename(p)[4:-4]
        if rows:
            b = min(range(len(rows)), key=lambda i: rows[i]['vloss'])
            avg = sum(r['sec'] for r in rows[-5:]) / len(rows[-5:])
            lines.append(f"{tag:10s} {len(rows):4d} {rows[b]['vloss']:10.4f} {b + 1:4d} {len(rows) - 1 - b:3d}/{PATIENCE} {avg:5.0f}  {status}")
        else:
            lines.append(f"{tag:10s} {'-':>4s} {'':10s} {'':4s} {'':6s} {'':5s}  {status} (starting / loading data)")

    if logs:  # epoch table for the most recent run
        p = logs[-1]
        rows, status = parse_cnn(p)
        tag = os.path.basename(p)[4:-4]
        lines += ['', f'Latest run: {tag}  ({status})']
        if rows:
            shown = rows[-a.last:]
            lo, hi = min(r['vloss'] for r in shown), max(r['vloss'] for r in shown)
            best = min(r['vloss'] for r in rows)
            lines.append(f"{'epoch':>5s} {'loss':>7s} {'acc':>7s} {'val_loss':>8s} {'val_acc':>7s} {'sec':>4s}  val_loss")
            start = len(rows) - len(shown)
            for i, r in enumerate(shown, start + 1):
                mark = ' <- best' if r['vloss'] == best else ''
                lines.append(f"{i:5d} {r['loss']:7.4f} {r['acc']:7.4f} {r['vloss']:8.4f} {r['vacc']:7.4f} {r['sec']:4d}  "
                             f"{bar(r['vloss'], lo, hi)}{mark}")
            if status == 'running':
                since = time.time() - os.path.getmtime(p)
                lines.append(f"epoch {len(rows) + 1} in progress: {since:.0f}s elapsed (avg {rows[-1]['sec']}s/epoch)")

    lines += ['', trials('stack_trials.jsonl'), trials('mfe_trials.jsonl')]
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=int, default=10)
    ap.add_argument('--run', help='CNN tag to show in the epoch table (default: most recent)')
    ap.add_argument('--last', type=int, default=15, help='epochs to show in the table')
    a = ap.parse_args()
    if a.once:
        print(show(a))
    else:
        os.system('')  # enables ANSI clear-screen codes in the classic Windows console
        try:
            while True:
                out = show(a)
                print('\033[2J\033[H' + out, flush=True)  # clear screen + home
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass
