import os
import lm_eval

pkgs_installed_dir = os.path.dirname(os.path.dirname(lm_eval.__file__))
alter_dir = os.path.dirname(__file__)

for root, _, files in os.walk(alter_dir):
    for filename in files:
        if filename not in ("base.py", "evaluator.py"):
            continue

        cur_file = os.path.abspath(os.path.join(root, filename))
        rel_path = cur_file.split('flexgen/')[-1]
        old_file = os.path.join(pkgs_installed_dir, rel_path)

        if not os.path.exists(old_file) and not os.path.islink(old_file):
            print(f'未实现: {old_file}')
            continue

        if os.path.islink(old_file):
            os.remove(old_file)
            os.symlink(cur_file, old_file)
            print(f'√ 更新软连接成功 {old_file} -> {cur_file}')
            continue

        backup_file = os.path.join(
            os.path.dirname(old_file),
            os.path.basename(old_file).split('.')[0] + '_bkup.py',
        )
        if not os.path.exists(backup_file):
            os.rename(old_file, backup_file)
            print(f'-> 已备份原文件为 {backup_file}')
        else:
            os.remove(old_file)
            print(f'-> 备份文件已存在，移除原文件 {old_file}')

        os.symlink(cur_file, old_file)
        print(f'√ 创建软连接成功 {old_file} -> {cur_file}')
