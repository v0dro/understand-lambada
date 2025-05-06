# understand-lambada

[日本語 README](README-ja.md)

Repo with code for gaining a deeper understanding of the LAMBADA dataset. The accompanying blog post for this repo -- that explains everything and has lots of benchmarks -- can be found [here](https://open.substack.com/pub/v0dro/p/understanding-long-context-information?r=9vifl&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true).

Please log into your hugging face account using the [hugging face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and request access for [Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) in order to smoothly run all the tests in this repo.

This code is a step-by-step walkthrough for inference and fine tuning with LAMBADA. A summary of the files in this repo is as follows:
1. `1_load_lambada.py`
    - Load LAMBADA from hugging face and show a graph of the distribution of training, test and validation data sets.
2. `2_analyse_lambada.py`
    - Analyse various parts of the dataset and print out some statistical data.
3. `3_load_test_sample.py`
    - Show the contents of a test sample.
4. `4_run_model_forward.py`
    - Call the `forward()` method for inference and print the cross-entropy loss.
5. `5_run_model_generate.py`
    - Call the `generate()` method for generating tokens and print the predicted token.
6. `6_select_training_dataset.py`
    - Select an entry from the training dataset to use in the fine tuning process.
7. `7_build_dataloader.py`
    - Load the selected data into a `DataLoader()` and write a basic iterator for fine tuning.
8. `8_finetune_lambada.py`
    - Use the data and tokenized strings generated above for fine tuning the dataset.
9. `9_finetune_and_inference_lambada.py`
    - Fine tune and use the fine tuned model for inference.