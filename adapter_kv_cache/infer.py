class ModelConfig:
    special_token_table: list=[]
    punctuation_table: list=[]
    locality_length : int = 10
    top_frequency: int = 10


def init_module():

    # load tokenizer

    # load model

    pass

def load_weight():
    pass

def create_test_prompts():
    input_token_ids=[1,2,3,4,5,6,7,8,9]

    pass

def run_infer():
    pass


def main():
    module = init_module()
    lora_path = load_weight(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)
    run_infer(engine, test_prompts)


if __name__ == '__main__':
    main()