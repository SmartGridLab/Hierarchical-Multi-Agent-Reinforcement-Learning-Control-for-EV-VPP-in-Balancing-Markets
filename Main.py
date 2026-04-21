"""
Main.py — プロジェクトのエントリーポイント

training.train モジュールの train() を呼び出して学習を開始する。
実行: python Main.py
"""


from training.train import (
train ,
test ,
set_env_seed ,
create_model_directory ,
sample_episode_demand_strict ,
launch_tensorboard ,
signal_handler ,
)

if __name__ =="__main__":
    # 学習を実行し、エージェント・報酬履歴・性能指標・エピソードデータ・作業ディレクトリを受け取る
    ag ,all_rewards ,perf ,ep_data ,work_dir =train ()
