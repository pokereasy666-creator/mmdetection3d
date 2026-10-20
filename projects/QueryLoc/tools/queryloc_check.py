#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consistency checks for exec doc #3 (QueryLoc).

s0ppp     S0''' (before every run): CONFIG / CKPT md5 and CFG_OPTS equal to
          the exec doc #2 E0 runtime record; projects/BEVFusion equal to the
          82c2b4e manifest (source DIFF / extra source file -> FAIL, .md DIFF
          -> WARN, build products ignored; same policy as exec doc #1 S0a);
          tools/query_diag equal to exec doc #2 step 2 (no change allowed
          yet); projects/QueryLoc equal to code_md5.json; sync gate (ASCII,
          ast for py3.8, bash -n) on projects/QueryLoc.
register  dev side only: (re)write projects/QueryLoc/code_md5.json.
Pure stdlib; no torch.
"""
import argparse
import ast
import hashlib
import json
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
QL_ROOT = osp.dirname(HERE)
REGISTRY = 'code_md5.json'
CODE_EXT = ('.py', '.sh', '.json')
SRC_EXT = ('.py', '.pyx', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp',
           '.json', '.yaml', '.yml', '.sh')
IGNORE_DIRS = ('__pycache__', 'build', '.eggs')


def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_version_line(path='VERSION'):
    try:
        with open(path) as f:
            return f.readline().strip() or 'unknown'
    except (OSError, UnicodeDecodeError):
        return 'unknown'


def gate_one(path):
    with open(path, 'rb') as f:
        raw = f.read()
    if any(b >= 128 for b in raw):
        return 'non-ASCII'
    if path.endswith('.py'):
        try:
            ast.parse(raw.decode(), feature_version=(3, 8))
        except SyntaxError as e:
            return 'ast: %s' % e
    elif path.endswith('.sh'):
        r = subprocess.run(['bash', '-n', path], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        if r.returncode:
            return 'bash -n: %s' % r.stdout.decode(errors='replace')
    elif path.endswith('.json'):
        try:
            json.loads(raw.decode())
        except ValueError as e:
            return 'json: %s' % e
    return ''


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in IGNORE_DIRS and
                       not d.endswith('.egg-info'))
        for n in sorted(fn):
            yield osp.join(dp, n)


def code_files(ql_root):
    out = {}
    for p in walk(ql_root):
        rel = osp.relpath(p, ql_root)
        if rel.endswith(CODE_EXT) and rel != REGISTRY:
            out[rel] = file_md5(p)
    return out


def cmd_register(args):
    reg = code_files(args.ql_root)
    path = osp.join(args.ql_root, REGISTRY)
    with open(path, 'w') as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write('\n')
    print('[queryloc] wrote %s (%d files)' % (path, len(reg)))
    return 0


def cmd_s0ppp(args):
    root = args.root
    L = ["S0''' consistency (exec doc #3) at %s" % osp.abspath(root)]
    fail, warn = [], []
    with open(args.ref_json) as f:
        ref = json.load(f).get('runtime', {})
    cm, km = file_md5(args.config), file_md5(args.ckpt)
    opts = os.environ.get('CFG_OPTS', '')
    for name, now, want in (('CONFIG md5', cm, ref.get('config_md5')),
                            ('CKPT md5', km, ref.get('checkpoint_md5')),
                            ('CFG_OPTS', opts, ref.get('cfg_opts_env'))):
        ok = now == want
        L.append('  %-10s %s  (#2 E0: %s)  %s'
                 % (name, now, want, 'OK' if ok else 'DIFF'))
        if not ok:
            fail.append('%s differs from exec doc #2 E0' % name)

    with open(osp.join(root, args.manifest)) as f:
        tree = json.load(f)['tree']
    bev = osp.join(root, 'projects', 'BEVFusion')
    n_ok = 0
    for rel, want in sorted(tree.items()):
        p = osp.join(root, rel)
        if not osp.isfile(p):
            fail.append('projects/BEVFusion: %s MISSING' % rel)
        elif file_md5(p) != want:
            (warn if rel.endswith('.md') else fail).append(
                'projects/BEVFusion: %s DIFF' % rel)
        else:
            n_ok += 1
    for p in walk(bev):
        rel = osp.relpath(p, root).replace(os.sep, '/')
        if rel in tree or rel.endswith(('.so', '.pyc', '.o')):
            continue
        (fail if rel.endswith(SRC_EXT) else warn).append(
            'projects/BEVFusion: %s EXTRA (not in the 82c2b4e manifest)'
            % rel)
    L.append('  projects/BEVFusion: %d / %d manifest files identical'
             % (n_ok, len(tree)))

    qd = osp.join(root, 'tools', 'query_diag')
    ref_qd = ref.get('diag_files_md5') or {}
    cur_qd = {f: file_md5(osp.join(qd, f)) for f in sorted(os.listdir(qd))
              if f.endswith(CODE_EXT)}
    qd_diff = sorted(f for f in set(ref_qd) | set(cur_qd)
                     if ref_qd.get(f) != cur_qd.get(f))
    for f in qd_diff:
        fail.append('tools/query_diag/%s %s since exec doc #2 step 2' % (
            f, 'NEW' if f not in ref_qd else
            ('GONE' if f not in cur_qd else 'CHANGED')))
    L.append('  tools/query_diag: %d files, %s'
             % (len(cur_qd), 'identical to exec doc #2 step 2'
                if not qd_diff else '%d differ' % len(qd_diff)))

    ql = osp.join(root, 'projects', 'QueryLoc')
    reg_path = osp.join(ql, REGISTRY)
    reg = json.load(open(reg_path)) if osp.isfile(reg_path) else {}
    if not reg:
        fail.append('projects/QueryLoc/%s missing or empty' % REGISTRY)
    cur = code_files(ql)
    for rel in sorted(set(reg) | set(cur)):
        if reg.get(rel) != cur.get(rel):
            fail.append('projects/QueryLoc/%s %s vs %s' % (
                rel, 'NEW' if rel not in reg else
                ('GONE' if rel not in cur else 'CHANGED'), REGISTRY))
    gates = {rel: gate_one(osp.join(ql, rel)) for rel in cur}
    gates[REGISTRY] = gate_one(reg_path) if osp.isfile(reg_path) else \
        'missing'
    for rel, e in sorted(gates.items()):
        if e:
            fail.append('sync gate: projects/QueryLoc/%s %s' % (rel, e))
    L.append('  projects/QueryLoc: %d files vs %s; sync gate %s'
             % (len(cur), REGISTRY,
                'all OK' if not any(gates.values()) else 'FAILED'))
    for rel in sorted(cur):
        L.append('    %-40s %s' % (rel, cur[rel]))
    for x in warn:
        L.append('  WARN ' + x)
    for x in fail:
        L.append('  FAIL ' + x)
    res = {'check': "S0'''", 'result': 'FAIL' if fail else 'PASS',
           'fail': fail, 'warn': warn, 'config_md5': cm,
           'checkpoint_md5': km, 'cfg_opts_env': opts,
           'queryloc_md5': cur, 'gate': gates,
           'runtime': {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'repo_version': read_version_line(
                           osp.join(root, 'VERSION'))}}
    L.append("S0'''  %s%s" % (res['result'],
                              '  (%d WARN)' % len(warn) if warn else ''))
    text = '\n'.join(L)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 's0ppp_check.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(args.out_dir, 's0ppp_check.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('[queryloc] saved %s/s0ppp_check.{txt,json}' % args.out_dir)
    return 0 if not fail else 1


def main():
    p = argparse.ArgumentParser(description='exec doc #3 checks')
    sub = p.add_subparsers(dest='cmd')
    s = sub.add_parser('s0ppp')
    s.add_argument('--root', default='.')
    s.add_argument('--config', required=True)
    s.add_argument('--ckpt', required=True)
    s.add_argument('--ref-json', default='work_dirs/diag/exec2/'
                   'oracle_e0_base/oracle_e0_base.json')
    s.add_argument('--manifest',
                   default='tools/query_diag/s0_manifest_82c2b4e.json')
    s.add_argument('--out-dir', default='work_dirs/queryloc/s0ppp')
    r = sub.add_parser('register')
    r.add_argument('--ql-root', default=QL_ROOT)
    args = p.parse_args()
    if args.cmd == 's0ppp':
        return cmd_s0ppp(args)
    if args.cmd == 'register':
        return cmd_register(args)
    p.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
