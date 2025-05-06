LAMBADAデータセットをより深く理解するためのコードが含まれたリポジトリです。このリポジトリに対応したブログ記事（すべてを説明し、多くのベンチマークを掲載しています）は[こちら](https://open.substack.com/pub/v0dro/p/understanding-long-context-information?r=9vifl&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)からご覧いただけます。

[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)を使用してHugging Faceアカウントにログインし、[Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)へのアクセスをリクエストしてください。これにより、このリポジトリ内のすべてのテストをスムーズに実行できます。

このコードは、LAMBADAを使った推論およびファインチューニングのステップバイステップの解説です。このリポジトリに含まれるファイルの概要は以下の通りです：

1. `1_load_lambada.py`  
    - Hugging FaceからLAMBADAを読み込み、トレーニング・テスト・検証データセットの分布グラフを表示します。
2. `2_analyse_lambada.py`  
    - データセットのさまざまな部分を分析し、統計データを出力します。
3. `3_load_test_sample.py`  
    - テストサンプルの内容を表示します。
4. `4_run_model_forward.py`  
    - 推論用に`forward()`メソッドを呼び出し、クロスエントロピー損失を表示します。
5. `5_run_model_generate.py`  
    - トークン生成のために`generate()`メソッドを呼び出し、予測されたトークンを表示します。
6. `6_select_training_dataset.py`  
    - ファインチューニングで使用するためにトレーニングデータセットからエントリを選択します。
7. `7_build_dataloader.py`  
    - 選択したデータを`DataLoader()`に読み込み、ファインチューニング用の基本的なイテレータを作成します。
8. `8_finetune_lambada.py`  
    - 上記で生成されたデータとトークン化された文字列を使用してデータセットをファインチューニングします。
9. `9_finetune_and_inference_lambada.py`  
    - モデルをファインチューニングし、そのモデルを使って推論を行います。