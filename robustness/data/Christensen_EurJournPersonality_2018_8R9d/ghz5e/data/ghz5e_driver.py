import json
import subprocess
import sys


def run_cmd(cmd):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(res.returncode)
    return res.stdout, res.stderr


def main():
    base_cmd = [
        'python', '/workspace/data/ghz5e_analysis__py.py',
        '--data_file', '/app/data/FINAL fluency.csv',
        '--open_file', '/app/data/FINAL open.csv',
        '--id_col', 'id',
        '--openness_col', 'LATENT',
        '--response_prefix', 'vf_an_',
        '--boot_iters', '300'
    ]

    # Task1
    cmd1 = base_cmd + ['--task', 'Task1']
    out1, err1 = run_cmd(cmd1)
    try:
        res1 = json.loads(out1)
    except Exception as e:
        sys.stderr.write('Failed to parse Task1 JSON: ' + str(e) + '\n')
        sys.stderr.write(out1 + '\n')
        raise

    # Task2 (90% retention)
    cmd2 = base_cmd + ['--task', 'Task2', '--bootstrap_prop', '0.9']
    out2, err2 = run_cmd(cmd2)
    try:
        res2 = json.loads(out2)
    except Exception as e:
        sys.stderr.write('Failed to parse Task2 JSON: ' + str(e) + '\n')
        sys.stderr.write(out2 + '\n')
        raise

    combined = {
        'Task1': res1,
        'Task2': res2
    }

    with open('/app/data/execution_result.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print('Wrote /app/data/execution_result.json with Task1 and Task2 results')


if __name__ == '__main__':
    main()
